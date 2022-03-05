import os
import glob
from typing import Dict

from allennlp.data import DatasetReader, Instance, Tokenizer
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer


@DatasetReader.register("style_transfer_pretrain_reader")
class StyleTransferPretrainReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        min_length: int = 16,
        max_length: int = 480,
        cache_files: bool = True,
        debug: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.min_length = min_length
        self.max_length = max_length

        if debug:
            self._instance_counter = 0
        self.style_to_token = dict()
        self.token_to_style = dict()

    def _read(self, file_path: str) -> Instance:
        author_paths = self._parse_file_path(file_path)
        for i, (author, path) in enumerate(author_paths.items()):
            style_token = f"[unused{i+1}]"
            self.style_to_token[author] = style_token
            self.token_to_style[style_token] = author
            docs = glob.glob(os.path.join(path, "*.txt"))
            for doc in docs:
                if getattr(self, "_instance_counter", -1) > 20:
                    self._instance_counter = 0
                    break
                with open(doc, 'r') as file:
                    content = file.read()
                    if len(content) < self.min_length or len(content) > self.max_length - 2:
                        continue
                    yield self.text_to_instance(content, author)

    def text_to_instance(self, text: str, author: str) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        # print(text)
        # while len(tokens) > self.max_length:
        #     for end, token in enumerate(reversed(tokens)):
        #         if end == 0:
        #             continue
        #         if token.text in ['。', '！', '？', '”']:
        #             tokens = tokens[: len(tokens) - end]
        #             break
        # assert len(tokens) <= self.max_length, text

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

    def _adjust_length(self, tokens):
        pass
