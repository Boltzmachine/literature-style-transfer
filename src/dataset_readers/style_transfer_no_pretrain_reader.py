import os
import glob
from typing import Dict, Optional

from allennlp.data import DatasetReader, Instance, Tokenizer
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer


@DatasetReader.register("style_transfer_no_pretrain_reader")
class StyleTransferNoPretrainReader(DatasetReader):
    def __init__(
        self,
        tokenizer=None,
        token_indexers: Dict[str, TokenIndexer] = None,
        min_length: int = 8,
        max_length: int = 512,
        cache_files: bool = True,
        debug: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or SpacyTokenizer("zh_core_web_sm")
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.min_length = min_length
        self.max_length = max_length

        if debug:
            self._instance_counter = 0

    def _read(self, file_path: str) -> Instance:
        author_paths = self._parse_file_path(file_path)
        for i, (author, path) in enumerate(author_paths.items()):
            docs = glob.glob(os.path.join(path, "*.txt"))
            for doc in docs:
                if getattr(self, "_instance_counter", -1) > 20:
                    self._instance_counter = 0
                    break
                with open(doc, 'r') as file:
                    content = file.read()
                    if len(content) < self.min_length:
                        continue
                    yield self.text_to_instance(content, author)

    def text_to_instance(self, text: str, author: str) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:self.max_length]  # TODO: more complicated

        text_field = TextField(tokens, token_indexers=self.token_indexers)
        style_field = LabelField(author, label_namespace="style_labels")

        fields = {
            "text": text_field,
            "style": style_field,
        }

        if hasattr(self, "_instance_counter"):
            self._instance_counter += 1

        return Instance(fields)

    def _parse_file_path(self, file_path: str) -> Dict:
        dirs = file_path.split('/')
        for i, dir in enumerate(dirs):
            if dir:
                if dir[0] == '{' and dir[-1] == '}':
                    dir = dir[1:-1]
                    authors = dir.split(',')
                    break
        authors = [author.strip() for author in authors]
        author_paths = {author: os.path.join(*dirs[:i], author, *dirs[i + 1:]) for author in authors}
        return author_paths
