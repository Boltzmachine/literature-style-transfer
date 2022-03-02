from dis import dis
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import RegularizerApplicator
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.data import TextFieldTensors
from allennlp.nn.parallel import DdpAccelerator

from transformers import BertModel, BartForConditionalGeneration
from .cpt import CPTDiscriminator, CPTGenerator, build_cpt_generator_and_discriminator
from .modeling_cpt import CPTModel


def get_soft_tokens_from_probs(probs, embedding_layer):
    soft_tokens = probs @ embedding_layer.weight
    return soft_tokens


def get_soft_tokens_from_logits(logits, embedding_layer):
    probs = torch.softmax(logits, dim=-1)
    return get_soft_tokens_from_probs(probs, embedding_layer)


def get_attention_mask_from_logits(logits, eos_token_id):
    tokens = logits.argmax(-1)
    max_length = logits.size(1) + 1  # for style token
    lengths = torch.cumsum(tokens == eos_token_id, -1)
    lengths = (lengths == 0).sum(-1)
    lengths = lengths + 1  # for eos token
    lengths = lengths + 1  # for style token
    mask = (torch.arange(max_length, device=lengths.device)[
            None, :] < lengths[:, None]).float()  # (1, max_length) < (lengths, 1)
    return mask


@Model.register("soft_tokens_generator")
class SoftTokensGenerator(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model
    ):
        super().__init__(vocab)
        self.model = model

        self.eos_token_id = model.config.eos_token_id

    def forward(
        self,
        src_input: TextFieldTensors,
        source_style: torch.IntTensor,
        target_style: torch.IntTensor,
        tgt_input: TextFieldTensors = None,
    ):
        input_ids = src_input['tokens'].get('token_ids')
        attention_mask = src_input['tokens']['mask']
        inputs_embeds = src_input['tokens'].get('token_embeds')

        decoder_input_ids = None if tgt_input is None else tgt_input['tokens']['token_ids']
        decoder_attention_mask = None if tgt_input is None else tgt_input['tokens']['mask'].float()

        input_len = input_ids.size(1) if input_ids is not None else inputs_embeds.size(1)
        input_ids, inputs_embeds = self._add_style_token(input_ids, inputs_embeds, style=source_style)

        encoder = self.model.get_encoder()
        encoder_outputs = encoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        generated_outputs = self.generate_token_by_token(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            begin_token=target_style,
            max_len=input_len,
            return_hard=True
        )
        # ref_tokens = self.model.generate(input_ids, num_beams=1, do_sample=False, max_length=512)

        logits = generated_outputs['logits']

        return logits

    def generate_token_by_token(
        self,
        encoder_outputs,
        attention_mask,
        begin_token=None,
        max_len=512,
        return_hard=False,
    ):
        # logits_processor = LogitsProcessorList([
        #     NoRepeatNGramLogitsProcessor(3),
        #     MinLengthLogitsProcessor(0, 102),
        #     ForcedEOSTokenLogitsProcessor(20, 102)
        # ])
        # prepare for inputs
        batch_size, seq_len, hidden_dim = encoder_outputs.last_hidden_state.size()
        device = encoder_outputs.last_hidden_state.device
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # pad_token_embed = self.model.get_input_embeddings()(torch.tensor(self.pad_token_id, device=device))

        decoder_input_ids = torch.ones(batch_size, 1, dtype=torch.long, device=device) * \
            (self.eos_token_id if begin_token is None else begin_token)
        decoder_inputs_embeds = self.model.get_input_embeddings()(decoder_input_ids)
        past_key_values = None

        generated_soft_tokens = []
        generated_hard_tokens = [] if return_hard else None
        generated_logits = []
        # begin generation
        # while True:
        for _ in range(max_len):
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                # decoder_input_ids=decoder_input_ids,
                decoder_inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :]
            # next_token_logits = logits_processor(torch.stack(generated_hard_tokens, dim=1), next_token_logits)
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            # next_token_id[finished] = self.pad_token_id

            next_token_embeds = get_soft_tokens_from_logits(next_token_logits, self.model.get_input_embeddings())
            # next_token_embeds = self.model.get_input_embeddings()(next_token_id)
            # next_token_embeds[finished] = pad_token_embed

            if return_hard:
                generated_hard_tokens.append(next_token_id)
            generated_soft_tokens.append(next_token_embeds)
            generated_logits.append(next_token_logits)
            # decoder_inputs_embeds = torch.cat([decoder_inputs_embeds, next_token_embeds], dim=1)
            decoder_inputs_embeds = next_token_embeds.unsqueeze(1)

            finished |= next_token_id == self.eos_token_id
            # if finished.all() or len(generated_soft_tokens) >= 512:
            #     break

        generated_soft_tokens = torch.stack(generated_soft_tokens, dim=1)
        generated_logits = torch.stack(generated_logits, dim=1)
        if return_hard:
            generated_hard_tokens = torch.stack(generated_hard_tokens, dim=1)

        return {
            "soft_tokens": generated_soft_tokens,
            "logits": generated_logits,
            "hard_tokens": generated_hard_tokens,
        }

    def _add_style_token(self, input_ids, inputs_embeds, style):
        # use [unused*] token
        style = (style + 1).unsqueeze(1)
        assert not (input_ids is not None and inputs_embeds is not None)
        if input_ids is not None:
            input_ids = torch.cat((style, input_ids), dim=1)
        if inputs_embeds is not None:
            style_embedding = self.model.get_input_embeddings()(style)
            inputs_embeds = torch.cat((style_embedding, inputs_embeds), dim=1)

        return input_ids, inputs_embeds

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        source_style,
        target_style,
    ):
        input_ids, _ = self._add_style_token(input_ids, None, style=source_style)
        return self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=target_style.unsqueeze(-1),
            num_beams=3,
            max_length=512,
        )


@Model.register("style_discriminator")
class StyleDiscriminator(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            model,
            regularizer: RegularizerApplicator = None,
            serialization_dir: Optional[str] = None,
            ddp_accelerator: Optional[DdpAccelerator] = None
    ) -> None:
        super().__init__(vocab, regularizer, serialization_dir, ddp_accelerator)
        self.model = model
        self.head = torch.nn.Linear(self.model.config.hidden_size, vocab.get_vocab_size("style_labels") + 1)

    def forward(
        self,
        inputs_embeds,
        attention_mask,
        cls_token_id=None
    ):
        if cls_token_id is not None:
            inputs_embeds = self._add_cls_token(inputs_embeds, cls_token_id)
        context = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        pooled_output = context.last_hidden_state[:, 1]
        logits = self.head(pooled_output)
        return logits

    def _add_cls_token(self, inputs_embeds, cls_token_id):
        to_add = torch.ones(inputs_embeds.size(0), 1, dtype=torch.long, device=inputs_embeds.device) * cls_token_id
        cls_embed = self.model.get_input_embeddings()(to_add)
        inputs_embeds = torch.cat([cls_embed, inputs_embeds], dim=1)
        return inputs_embeds


@Model.register("style_transferer")
class StyleTransferer(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        regularizer: RegularizerApplicator = None,
        serialization_dir: Optional[str] = None,
    ):
        super().__init__(vocab, regularizer, serialization_dir)

        cpt_model = CPTModel.from_pretrained(model_name)
        generator, discriminator = build_cpt_generator_and_discriminator(cpt_model=cpt_model)
        self.generator = SoftTokensGenerator(vocab, generator)
        self.discriminator = StyleDiscriminator(vocab, discriminator)

        self._assert_special_token_id_and_set_attr('pad_token_id')
        self._assert_special_token_id_and_set_attr('eos_token_id')
        self._assert_special_token_id_and_set_attr('bos_token_id')

    def forward(
        self,
        text: TextFieldTensors,
        style: torch.IntTensor,
    ):
        text['tokens']['token_ids'] = text['tokens']['token_ids'][:, 1:]  # remove [CLS] token
        org_token_ids = text['tokens']['token_ids']

        # sample style
        _style = 1 - style

        # generation step
        # generate self style
        gen_self_logits = self.generator(
            src_input=text,
            source_style=style,
            target_style=style
        )

        rec_loss = F.cross_entropy(gen_self_logits.transpose(-1, -2),
                                   org_token_ids, ignore_index=self.pad_token_id)

        # generate another style
        gen_othr_logits = self.generator(
            src_input=text,
            source_style=style,
            target_style=_style,
        )
        gen_othr_soft_tokens = get_soft_tokens_from_logits(
            gen_othr_logits, self.generator.model.get_input_embeddings())
        gen_othr_attention_mask = get_attention_mask_from_logits(gen_othr_logits, self.eos_token_id)
        gen_cyc_logits = self.generator(
            src_input={'tokens': {
                'token_embeds': gen_othr_soft_tokens,
                'mask': gen_othr_attention_mask
            }},
            source_style=_style,
            target_style=style,
        )
        cyc_loss = F.cross_entropy(gen_cyc_logits.transpose(-1, -2),
                                   org_token_ids, ignore_index=self.pad_token_id)

        # AFAIK bert-base-chinese vocab is the same as fnlp/bart-base-chinese
        dis_othr_soft_tokens = get_soft_tokens_from_logits(
            gen_othr_logits, self.discriminator.model.get_input_embeddings())

        style_pred = self.discriminator(
            inputs_embeds=dis_othr_soft_tokens,
            attention_mask=gen_othr_attention_mask,
            cls_token_id=self.bos_token_id
        )
        style_loss = F.cross_entropy(style_pred, style + 1)

        gen_loss = 0.25 * rec_loss + 0.5 * cyc_loss + style_loss

        # discriminator step
        fake_labels = torch.zeros_like(style)
        disc_pred = self.discriminator(
            inputs_embeds=dis_othr_soft_tokens.detach(),
            attention_mask=gen_othr_attention_mask,
            cls_token_id=self.bos_token_id,
        )
        disc_loss = F.cross_entropy(disc_pred, fake_labels)

        outputs = {
            'gen_loss': gen_loss,
            'disc_loss': disc_loss,
        }

        return outputs

    def _assert_special_token_id_and_set_attr(self, token: str):
        gen_id = getattr(self.generator.model.config, token)
        dis_id = getattr(self.discriminator.model.config, token)
        if gen_id is not None and dis_id is not None:
            assert gen_id == dis_id
        setattr(self, token, gen_id)

    @torch.no_grad()
    def generate(
        self,
        text: TextFieldTensors,
        style: torch.IntTensor,
    ):
        text['tokens']['token_ids'] = text['tokens']['token_ids'][:, 1:]  # remove [CLS] token
        input_ids = text['tokens']['token_ids']

        return self.generator.generate(
            input_ids=input_ids,
            attention_mask=text['tokens']['mask'],
            source_style=style,
            target_style=1 - style,
        )
