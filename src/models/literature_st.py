from typing import Optional

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import RegularizerApplicator


@Model.register("literature_st")
class LiteratureST(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        pretrained_model: str,
        regularizer: RegularizerApplicator = None,
        serialization_dir: Optional[str] = None,
    ):
        super().__init__(vocab, regularizer, serialization_dir)

        self.seq2seq = None

    def forward(
        self,
        input_ids,
    ):
        tgt = self.seq2seq(input_ids)
        src = self.seq2seq(tgt)
