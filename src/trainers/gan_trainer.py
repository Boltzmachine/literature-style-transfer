import datetime
import glob
import logging
import math
import os
import re
import time
import warnings
from typing import Optional, Union, List, Dict, Tuple, Any, Type
from itertools import cycle

import torch
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import OptState
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import util as common_util, Tqdm, Lazy
from allennlp.common.file_utils import hardlink_or_copy
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.models.model import Model
from allennlp.nn.parallel import DdpAccelerator, DdpWrappedModel, TorchDdpAccelerator
from allennlp.nn.util import dist_reduce_sum
from allennlp.training.callbacks import ConsoleLoggerCallback
from allennlp.training.callbacks.confidence_checks import ConfidenceChecksCallback
from allennlp.training.callbacks.backward import MixedPrecisionBackwardCallback
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers.momentum_scheduler import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer, TrainerCheckpoint
from allennlp.training.callbacks import TrainerCallback
from allennlp.training import util as training_util
from allennlp.data import TextFieldTensors

from transformers import BertTokenizer

import wandb

logger = logging.getLogger(__name__)


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


@Trainer.register("gan", constructor="from_partial_objects")
class GANTrainer(Trainer):
    def __init__(
        self,
        model: Model,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        patience: Optional[int] = None,
        validation_metric: Union[str, List[str]] = "-loss",
        validation_data_loader: DataLoader = None,
        num_epochs: int = 20,
        serialization_dir: Optional[Union[str, os.PathLike]] = None,
        checkpointer: Optional[Checkpointer] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: Union[float, bool] = False,
        grad_clipping: Optional[float] = None,
        # learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        # momentum_scheduler: Optional[MomentumScheduler] = None,
        moving_average: Optional[MovingAverage] = None,
        callbacks: List[TrainerCallback] = None,
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        enable_default_callbacks: bool = True,
        run_confidence_checks: bool = True,
        grad_scaling: bool = True,
        ddp_wrapped_model: Optional[DdpWrappedModel] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
        )

        self.model = model

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.data_loader = data_loader
        self.data_loader.set_target_device(cuda_device)
        self.validation_data_loader = validation_data_loader

        self.patience = patience
        self.num_epochs = num_epochs

        self._checkpointer = checkpointer

        os.environ['TOKENIZERS_PARALLELISM'] = "false"
        self.wandb_run = wandb.init(project="literature-style-transfer")

        self.eval_data = self._create_fixed_eval_loader(data_loader)
        self.tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")

    def train(self) -> Dict[str, Any]:
        try:
            self._maybe_restore_checkpoint()
        except RuntimeError as e:
            configuration_error = ConfigurationError(
                f"Could not recover training from the checkpoint in {self._serialization_dir}. "
                "Did you mean to output to a different serialization directory or delete the "
                "existing serialization directory?"
            )
            configuration_error.__cause__ = e
            raise configuration_error

        # Set default values in case of failure
        epoch = None
        metrics = None

        for epoch in range(self.num_epochs):
            metrics = self._train_epoch(epoch)

    def g_step(self, text, style, _style):
        # generation step
        # generate self style
        generator = self.model.generator
        discriminator = self.model.discriminator
        gen_self_logits = generator(
            src_input=text,
            source_style=style,
            target_style=style
        )

        rec_loss = F.cross_entropy(gen_self_logits.transpose(-1, -2),
                                   text['tokens']['token_ids'], ignore_index=self.model.pad_token_id)

        # generate another style
        gen_othr_logits = generator(
            src_input=text,
            source_style=style,
            target_style=_style,
        )
        gen_othr_soft_tokens = get_soft_tokens_from_logits(
            gen_othr_logits, generator.model.get_input_embeddings())
        gen_othr_attention_mask = get_attention_mask_from_logits(gen_othr_logits, self.model.eos_token_id)
        gen_cyc_logits = generator(
            src_input={'tokens': {
                'token_embeds': gen_othr_soft_tokens,
                'mask': gen_othr_attention_mask
            }},
            source_style=_style,
            target_style=style,
        )
        cyc_loss = F.cross_entropy(gen_cyc_logits.transpose(-1, -2),
                                   text['tokens']['token_ids'], ignore_index=self.model.pad_token_id)

        # AFAIK bert-base-chinese vocab is the same as fnlp/bart-base-chinese
        dis_othr_soft_tokens = get_soft_tokens_from_logits(
            gen_othr_logits, discriminator.model.get_input_embeddings())

        style_pred = discriminator(
            inputs_embeds=dis_othr_soft_tokens,
            attention_mask=gen_othr_attention_mask,
            cls_token_id=self.model.bos_token_id
        )
        style_loss = F.cross_entropy(style_pred, style + 1)

        gen_loss = 0.25 * rec_loss + 0.5 * cyc_loss + style_loss

        self.g_optimizer.zero_grad()
        gen_loss.backward()
        self.g_optimizer.step()

        return gen_loss, dis_othr_soft_tokens.detach(), gen_othr_attention_mask.detach()

    def d_step(self, style, dis_othr_soft_tokens, gen_othr_attention_mask):
        discriminator = self.model.discriminator
        # discriminator step
        fake_labels = torch.zeros_like(style)
        disc_pred = discriminator(
            inputs_embeds=dis_othr_soft_tokens,
            attention_mask=gen_othr_attention_mask,
            cls_token_id=self.model.bos_token_id,
        )
        disc_loss = F.cross_entropy(disc_pred, fake_labels)

        self.d_optimizer.zero_grad()
        disc_loss.backward()
        self.d_optimizer.step()

        return disc_loss

    def _run_batch(
        self,
        text: TextFieldTensors,
        style: torch.IntTensor,
    ):

        text['tokens']['token_ids'] = text['tokens']['token_ids'][:, 1:]  # remove [CLS] token
        org_token_ids = text['tokens']['token_ids']

        # sample style
        _style = 1 - style

        gen_loss, dis_othr_soft_tokens, gen_othr_attention_mask = self.g_step(text, style, _style)
        disc_loss = self.d_step(style, dis_othr_soft_tokens, gen_othr_attention_mask)

        outputs = {
            'gen_loss': gen_loss.item(),
            'disc_loss': disc_loss.item(),
        }

        return outputs

    def _train_epoch(self, epoch: int):
        _gen_loss = 0
        _disc_loss = 0
        num_completed_batches = 0
        self.model.train()
        for batch in Tqdm.tqdm(self.data_loader):
            output_dict = self._run_batch(**batch)
            gen_loss = output_dict['gen_loss']
            disc_loss = output_dict['disc_loss']

            _gen_loss += gen_loss
            _disc_loss += disc_loss
            num_completed_batches += 1

        self.model.eval()
        table = self._evaluation()
        self.wandb_run.log({
            "gen_loss": _gen_loss / num_completed_batches,
            "disc_loss": _disc_loss / num_completed_batches,
            f"text_{epoch}": table,
        })

        return None

    def _evaluation(self):
        columns = ["source_style", "source_text", "target_style", "target_text"]
        table = wandb.Table(columns=columns)
        vocab = self.model.vocab

        for data in self.eval_data:
            outputs = self.model.generate(**data)
            source_sents = self.tokenizer.batch_decode(data['text']['bart']['token_ids'], skip_special_tokens=True)
            source_styles = data['style'].tolist()
            target_styles = outputs[:, 1].tolist()
            target_sents = self.tokenizer.batch_decode(outputs[:, 1:], skip_special_tokens=True)
            for src_sty, src_txt, tgt_sty, tgt_txt in zip(source_styles, source_sents, target_styles, target_sents):
                tgt_sty = 1 - src_sty
                src_sty = vocab.get_token_from_index(src_sty, namespace='style_labels')
                src_txt = ''.join(src_txt.split())
                tgt_txt = ''.join(tgt_txt.split())
                table.add_data(src_sty, src_txt, tgt_sty, tgt_txt)

        return table

    def get_checkpoint_state(self) -> Optional[TrainerCheckpoint]:
        model_state = self.model.state_dict()
        trainer_state = {
            "epoch": self.epoch
        }
        return TrainerCallback(model_state, trainer_state)

    def _create_fixed_eval_loader(self, data_loader, size=50):
        eval_data = []
        for i, data in enumerate(data_loader):
            eval_data.append(data)
            if i >= size:
                break
        return eval_data

    def _maybe_restore_checkpoint(self):
        return

    @classmethod
    def from_partial_objects(
        cls,
        model: Model,
        serialization_dir: str,
        data_loader: DataLoader,
        validation_data_loader: DataLoader = None,
        local_rank: int = 0,
        patience: int = None,
        validation_metric: Union[str, List[str]] = "-loss",
        num_epochs: int = 20,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: Union[float, bool] = False,
        grad_clipping: float = None,
        distributed: bool = False,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        no_grad: List[str] = None,
        g_optimizer: Lazy[Optimizer] = Lazy(Optimizer.default),
        d_optimizer: Lazy[Optimizer] = Lazy(Optimizer.default),
        # learning_rate_scheduler: Lazy[LearningRateScheduler] = None,
        # momentum_scheduler: Lazy[MomentumScheduler] = None,
        moving_average: Lazy[MovingAverage] = None,
        checkpointer: Optional[Lazy[Checkpointer]] = Lazy(Checkpointer),
        callbacks: List[Lazy[TrainerCallback]] = None,
        enable_default_callbacks: bool = True,
        run_confidence_checks: bool = True,
        grad_scaling: bool = True,
        ddp_accelerator: Optional[DdpAccelerator] = None,
        **kwargs,
    ) -> Trainer:
        if cuda_device is None:
            from torch import cuda

            if cuda.device_count() > 0:
                cuda_device = 0
            else:
                cuda_device = -1

        check_for_gpu(cuda_device)
        # Need to wrap model with a DdpAccelerator ("Distributed data-parallel wrapper")
        # or move model to right device before initializing the optimizer.
        # Using DDP brings in a quirk wrt AllenNLP's `Model` interface and its
        # usage. A `Model` object is wrapped by `DdpAccelerator`, but assigning the wrapped model to `self.model`
        # will break the usages such as `Model.get_regularization_penalty`, `Model.get_metrics`, etc.
        # Hence a reference to Pytorch's object is maintained in the case of distributed training and in the
        # normal case, reference to `Model` is retained. This reference is only used in
        # these places: `model.__call__`, `model.train` and `model.eval`.
        ddp_wrapped_model: Optional[DdpWrappedModel] = None
        if distributed:
            if ddp_accelerator is None:
                ddp_accelerator = TorchDdpAccelerator(cuda_device=cuda_device)
            # DdpAccelerator will move the model to the right device(s).
            model, ddp_wrapped_model = ddp_accelerator.wrap_model(model)
        else:
            if cuda_device >= 0:
                model = model.cuda(cuda_device)

        pytorch_model = model if ddp_wrapped_model is None else ddp_wrapped_model.model

        if no_grad:
            for name, parameter in pytorch_model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)

        g_parameters = [[n, p] for n, p in pytorch_model.generator.named_parameters() if p.requires_grad]
        d_parameters = [[n, p] for n, p in pytorch_model.discriminator.named_parameters() if p.requires_grad]
        g_optimizer_ = g_optimizer.construct(model_parameters=g_parameters)
        d_optimizer_ = d_optimizer.construct(model_parameters=d_parameters)

        common_util.log_frozen_and_tunable_parameter_names(pytorch_model)

        batches_per_epoch: Optional[int]
        try:
            batches_per_epoch = len(data_loader)
            batches_per_epoch = math.ceil(batches_per_epoch / num_gradient_accumulation_steps)
        except TypeError:
            batches_per_epoch = None

        moving_average_ = (
            None if moving_average is None else moving_average.construct(parameters=parameters)
        )
        # learning_rate_scheduler_ = (
        #     None
        #     if learning_rate_scheduler is None
        #     else learning_rate_scheduler.construct(
        #         optimizer=optimizer_, num_epochs=num_epochs, num_steps_per_epoch=batches_per_epoch
        #     )
        # )
        # momentum_scheduler_ = (
        #     None
        #     if momentum_scheduler is None
        #     else momentum_scheduler.construct(optimizer=optimizer_)
        # )
        checkpointer_ = (
            None
            if checkpointer is None
            else checkpointer.construct(serialization_dir=serialization_dir)
        )

        callbacks_: List[TrainerCallback] = []
        for callback_ in callbacks or []:
            callbacks_.append(callback_.construct(serialization_dir=serialization_dir))

        return cls(
            model,
            g_optimizer=g_optimizer_,
            d_optimizer=d_optimizer_,
            data_loader=data_loader,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            checkpointer=checkpointer_,
            moving_average=moving_average_,
            callbacks=callbacks_,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            use_amp=use_amp,
            enable_default_callbacks=enable_default_callbacks,
            run_confidence_checks=run_confidence_checks,
            grad_scaling=grad_scaling,
            ddp_wrapped_model=ddp_wrapped_model,
            **kwargs,
        )
