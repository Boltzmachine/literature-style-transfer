from typing import Dict, Optional

import torch
import torch.nn.functional as F

from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import TextFieldTensors
from allennlp.nn import RegularizerApplicator
from allennlp.nn.parallel import DdpAccelerator
from allennlp.training.metrics import Metric


@Model.register("style_classifier")
class StyleClassifier(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            contextualizer: TextFieldEmbedder,
            metrics: Dict[str, Metric] = None,
            train_alone: bool = False,
            regularizer: RegularizerApplicator = None,
            serialization_dir: Optional[str] = None,
            ddp_accelerator: Optional[DdpAccelerator] = None
    ) -> None:
        super().__init__(vocab, regularizer, serialization_dir, ddp_accelerator)
        self.contextualizer = contextualizer
        self.head = torch.nn.Linear(self.contextualizer.get_output_dim(), vocab.get_vocab_size("labels"))

        self.train_alone = train_alone
        self.metrics = metrics

    def forward(
        self,
        text: TextFieldTensors,
        label: torch.IntTensor
    ):
        context = self.contextualizer(text)
        pooled_output = context[:, 1]
        logits = self.head(pooled_output)
        for metric in self.metrics.values():
            metric(logits, label)

        if self.train_alone:
            probs = F.softmax(logits, dim=-1)
            loss = F.cross_entropy(logits, label)

            output = {
                'loss': loss,
                "probs": probs
            }
        else:
            return logits

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {k: v.get_metric(reset) for k, v in self.metrics.items()}
