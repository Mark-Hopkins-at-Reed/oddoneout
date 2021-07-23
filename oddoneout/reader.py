import tempfile
import json
import torch
from typing import Dict, Iterable, List
from allennlp.data import DatasetReader, Instance, Vocabulary, TextFieldTensors
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
#import random
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


@DatasetReader.register('o3')
class O3Reader(DatasetReader):
    def __init__(
            self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            max_tokens: int = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.question_prefix = "Which does not belong ?"
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                fields = line.strip().split('\t')
                oddone, choices = fields[0], fields[1:]
                bullets = ["(A)", "(B)", "(C)", "(D)", "(E)"]
                choice_str = " ".join([" ".join(item) for item in zip(bullets, choices)])
                text = self.question_prefix + " " + choice_str
                text_field = TextField(self.tokenizer.tokenize(text),
                                       self.token_indexers)
                label_field = LabelField(oddone)
                fields = {'text': text_field, 'label': label_field}
                yield Instance(fields)


@Model.register("simple_classifier")
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(
        self, text: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


