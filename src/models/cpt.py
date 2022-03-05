from cmath import log
from typing import Optional
import allennlp

import torch
import torch.nn as nn
import transformers
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput

from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.nn import RegularizerApplicator
from allennlp.nn.parallel import DdpAccelerator

from transformers.generation_utils import GenerationMixin
from transformers.modeling_utils import ModuleUtilsMixin


def build_cpt_generator_and_discriminator(cpt_model):
    senc, udec, gdec = get_cpt_seperate(cpt_model)
    return (
        CPTGenerator(cpt_model.config, senc, gdec),
        CPTDiscriminator(cpt_model.config, senc, udec)
    )


@Model.register('cpt-generator')
class CPTGenerator(Model, GenerationMixin, ModuleUtilsMixin):
    def __init__(self,
                 config,
                 encoder,
                 decoder,
                 regularizer: RegularizerApplicator = None,
                 serialization_dir: Optional[str] = None,
                 ddp_accelerator: Optional[DdpAccelerator] = None) -> None:
        super().__init__(regularizer, serialization_dir, ddp_accelerator)
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

        self.lm_head = nn.Linear(config.d_model, self.encoder.embeddings.word_embeddings.num_embeddings)

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_encoder(self):
        class _Encoder(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, *args, **kwargs):
                kwargs['output_hidden_states'] = True
                return self.encoder(*args, **kwargs)
        return _Encoder(self.encoder)

    def forward(
        self,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        attention_mask=None,
        past_key_values=None,
        input_ids=None,
        decoder_inputs_embeds=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        output_attentions=True,
        output_hidden_states=False,
        use_cache=False,
        return_dict=False
    ):
        assert not ((input_ids is not None) and (inputs_embeds is not None))
        assert not ((decoder_input_ids is not None) and (decoder_inputs_embeds is not None))

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # token_type_ids=torch.ones_like(input_ids or inputs_embeds),
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        encoder_hidden_states = encoder_outputs.hidden_states[-1]
        decoder_outputs = self.decoder(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(decoder_outputs.last_hidden_state)

        return Seq2SeqLMOutput(
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
        )

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs=None,
        **model_kwargs,
    ):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            device = encoder_outputs.last_hidden_state.device
            encoder_outputs["hidden_states"] = tuple(h.index_select(0, expanded_return_idx.to(device))
                                                     for h in encoder_outputs["hidden_states"])
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


@Model.register('cpt-discriminator')
class CPTDiscriminator(Model):
    def __init__(self,
                 config,
                 encoder,
                 decoder,
                 regularizer: RegularizerApplicator = None,
                 serialization_dir: Optional[str] = None,
                 ddp_accelerator: Optional[DdpAccelerator] = None) -> None:
        super().__init__(regularizer, serialization_dir, ddp_accelerator)
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def forward(
        self,
        inputs_embeds,
        attention_mask,
        return_dict=True
    ):
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # output_attentions=True,
            return_dict=return_dict
        )

        outputs = self.decoder(
            hidden_states=encoder_outputs.last_hidden_state,
            attention_mask=attention_mask,
            return_dict=return_dict
        )

        return outputs


def get_cpt_seperate(cpt_model):
    senc = CPTSEnc(cpt_model)
    udec = CPTUDec(cpt_model)
    gdec = CPTGDec(cpt_model)

    return senc, udec, gdec


class HuggingfaceModel():
    def __init__(self, model):
        self.config = model.config
        self.get_extended_attention_mask = model.get_extended_attention_mask
        self.get_head_mask = model.get_head_mask


class CPTSEnc(nn.Module, HuggingfaceModel):
    def __init__(self, cpt_model) -> None:
        nn.Module.__init__(self)
        HuggingfaceModel.__init__(self, cpt_model)
        self.embeddings = cpt_model.encoder.embeddings
        layer = cpt_model.encoder.encoder.layer[:-cpt_model.num_decoder_layers]
        self.encoder = BertEncoder(self.config, layer)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output, ) + encoder_outputs[1:]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class CPTUDec(nn.Module, HuggingfaceModel):
    def __init__(self, cpt_model) -> None:
        nn.Module.__init__(self)
        HuggingfaceModel.__init__(self, cpt_model)
        layer = cpt_model.encoder.encoder.layer[-cpt_model.num_decoder_layers:]
        self.decoder = BertEncoder(cpt_model.config, layer)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):

        outputs = self.decoder(
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=outputs.last_hidden_state
        )


class CPTGDec(nn.Module, HuggingfaceModel):
    def __init__(self, cpt_model) -> None:
        nn.Module.__init__(self)
        HuggingfaceModel.__init__(self, cpt_model)
        self.config = cpt_model.config
        self.decoder = cpt_model.decoder

    def forward(
        self,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_head_mask=None,
        decoder_head_mask=None,
        decoder_inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=encoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.config = config
        self.layer = layer
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
