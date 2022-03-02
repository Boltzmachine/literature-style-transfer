import os
import glob
from typing import Dict

from allennlp.data import DatasetReader, Instance, Tokenizer
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer


@DatasetReader.register("style_classification_reader")
class StyleClassificationReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        min_length: int = 8,
        max_length: int = 512,
        cache_files: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.min_length = min_length
        self.max_length = max_length

    def _read(self, file_path: str) -> Instance:
        author_paths = self._parse_file_path(file_path)
        for author, path in author_paths.items():
            docs = glob.glob(os.path.join(path, "*.txt"))
            for doc in docs:
                with open(doc, 'r') as file:
                    content = file.read()
                    yield self.text_to_instance(content, author)

    def text_to_instance(self, text: str, author: str) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:self.max_length]  # TODO: more complicated
        text_field = TextField(tokens, token_indexers=self.token_indexers)
        label_field = LabelField(author)

        fields = {
            "text": text_field,
            "label": label_field
        }

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