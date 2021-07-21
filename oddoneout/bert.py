import sys
import torch
from transformers import BertTokenizer, BertModel, AlbertTokenizer, AlbertModel
from transformers import GPT2Tokenizer, GPT2Model, RobertaTokenizer, RobertaModel
from transformers import CamembertTokenizer, CamembertModel, T5Tokenizer, T5Model
from transformers import FlaubertTokenizer, FlaubertModel, BartTokenizer, BartModel
from transformers import XLNetTokenizer, XLNetModel, XLMTokenizer, XLMModel
from oddoneout.solver import solve_puzzles
from oddoneout.puzzle import read_ooo_puzzles_from_tsv
from sklearn.metrics.pairwise import cosine_similarity
from oddoneout.similarity import SimilarityScore


class MLMEncoder:

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, msg):
        inputs = self.tokenizer(msg, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state


class AlbertEncoder(MLMEncoder):

    def __init__(self, model='albert-base-v2'):
        super().__init__(AlbertTokenizer.from_pretrained(model),
                         AlbertModel.from_pretrained(model))


class BartEncoder(MLMEncoder):

    def __init__(self, model='facebook/bart-base'):
        super().__init__(BartTokenizer.from_pretrained(model),
                         BartModel.from_pretrained(model))


class BertEncoder(MLMEncoder):

    def __init__(self, model='bert-base-uncased'):
        super().__init__(BertTokenizer.from_pretrained(model),
                         BertModel.from_pretrained(model))


class CamembertEncoder(MLMEncoder):

    def __init__(self, model='camembert-base'):
        super().__init__(CamembertTokenizer.from_pretrained(model),
                         CamembertModel.from_pretrained(model))


class FlaubertEncoder(MLMEncoder):

    def __init__(self, model='flaubert-base-uncased'):
        super().__init__(FlaubertTokenizer.from_pretrained(model),
                         FlaubertModel.from_pretrained(model))


class GPT2Encoder(MLMEncoder):

    def __init__(self, model='gpt2'):
        super().__init__(GPT2Tokenizer.from_pretrained(model),
                         GPT2Model.from_pretrained(model))


class RobertaEncoder(MLMEncoder):

    def __init__(self, model='roberta-base'):
        super().__init__(RobertaTokenizer.from_pretrained(model),
                         RobertaModel.from_pretrained(model))


class T5Encoder(MLMEncoder):

    def __init__(self, model='t5-large'):
        super().__init__(T5Tokenizer.from_pretrained(model),
                         T5Model.from_pretrained(model))


class XLNetEncoder(MLMEncoder):

    def __init__(self, model='xlnet-base-cased'):
        super().__init__(XLNetTokenizer.from_pretrained(model),
                         XLNetModel.from_pretrained(model))


class XLMEncoder(MLMEncoder):

    def __init__(self, model='xlm-mlm-en-2048'):
        super().__init__(XLMTokenizer.from_pretrained(model),
                         XLMModel.from_pretrained(model))


def initialize_encoder(encoder_id):
    if 'albert' in encoder_id:
        return AlbertEncoder(encoder_id)
    elif 'camembert' in encoder_id:
        return CamembertEncoder(encoder_id)
    elif 'flaubert' in encoder_id:
        return FlaubertEncoder(encoder_id)
    elif 'gpt2' in encoder_id:
        return GPT2Encoder(encoder_id)
    elif 'roberta' in encoder_id:
        return RobertaEncoder(encoder_id)
    elif 't5' in encoder_id:
        return T5Encoder(encoder_id)
    elif 'xlm' in encoder_id:
        return XLMEncoder(encoder_id)
    elif 'xlnet' in encoder_id:
        return XLNetEncoder(encoder_id)
    elif 'bart' in encoder_id:
        return BartEncoder(encoder_id)
    else:
        return BertEncoder(encoder_id)


ENCODERS = ['albert-base-v2',
            'albert-large-v2'
            'albert-xlarge-v2',
            'bert-base-uncased',
            'bert-large-uncased',
            'bert-base-multilingual-uncased',
            'distilroberta-base',
            'facebook/bart-base',
            'flaubert/flaubert_base_uncased',
            'flaubert/flaubert_base_cased',
            'gpt2',
            'gpt2-medium',
            'roberta-base',
            'roberta-large',
            't5-large',
            'xlm-mlm-en-2048',
            'xlnet-base-cased',
            'xlnet-large-cased']


COMPRESSORS = {'all': lambda t: t[:, :, :].mean(dim=1),
               'zeroth': lambda t: t[:, 0:1, :].mean(dim=1),
               'first': lambda t: t[:, 1:2, :].mean(dim=1),
               'last': lambda t: t[:, -1:, :].mean(dim=1),
               'secondlast': lambda t: t[:, -2:-1, :].mean(dim=1),
               'trimmed': lambda t: t[:, 1:-1, :].mean(dim=1)
               }


class Vectorizer:

    def __init__(self, encoder, compressor):
        self.encoder = encoder
        self.compressor = compressor

    def __call__(self, msg):
        return self.compressor(self.encoder(msg))


class VectorSimilarity(SimilarityScore):

    def __init__(self, vectorizer):
        super().__init__()
        self.vectorizer = vectorizer

    def __call__(self, words):
        vecs = torch.cat([self.vectorizer(word) for word in words]).detach().numpy()
        sims = cosine_similarity(vecs, vecs).sum(axis=1)
        return [1.0/sim for sim in sims]

    def is_recognized(self, word):
        return True


def run_experiment(puzzles, encoder, compressor):
    try:
        vectorizer = Vectorizer(encoder, compressor)
        similarity = VectorSimilarity(vectorizer)
        (win, lose, tie) = solve_puzzles(puzzles, similarity)
        return win, lose, tie
    except ValueError:
        return "FAIL"


def run_experiments(ooo_filename, encoder_ids, compressor_ids):
    results = dict()
    puzzles = list(read_ooo_puzzles_from_tsv(ooo_filename))
    for encoder_id in encoder_ids:
        encoder = initialize_encoder(encoder_id)
        for compressor_id in compressor_ids:
            print('Running experiment ({}, {})'.format(encoder_id, compressor_id))
            compressor = COMPRESSORS[compressor_id]
            result = run_experiment(puzzles, encoder, compressor)
            results[(encoder_id, compressor_id)] = result
    return results


if __name__ == "__main__":
    puzzle_file = sys.argv[1]
    output_file = sys.argv[2]
    encoder_ids = ENCODERS
    compressor_ids = COMPRESSORS.keys()
    exp_results = run_experiments(puzzle_file, encoder_ids, compressor_ids)
    with open(output_file, 'w') as writer:
        for (enc_id, comp_id) in exp_results:
            results = exp_results[enc_id, comp_id]
            if results == "FAIL":
                writer.write(','.join([enc_id, comp_id, "FAIL"]) + "\n")
            else:
                writer.write(','.join([enc_id, comp_id] + [str(r) for r in results]) + "\n")
            print('({}, {}): {}'.format(enc_id, comp_id, results))

    # print("Correct: {}\nIncorrect: {}\nAbstained: {}".format(win, lose, tie))



