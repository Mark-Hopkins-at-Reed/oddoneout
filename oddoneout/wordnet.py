"""
import nltk
nltk.download('wordnet')
"""
import sys
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from oddoneout.taxonomy import Taxonomy, Specificity, TaxonomySimilarity
from oddoneout.solver import solve_puzzles
from oddoneout.puzzle import read_ooo_puzzles_from_tsv
from oddoneout.similarity import SimilarityScore


class WordnetTaxonomy(Taxonomy):

    def __init__(self):
        self.specificity = Specificity()
        self.num_insts = len(self.get_descendant_instances(self.get_root()))

    def is_instance(self, node):
        return len(wn.synsets(node)) > 0

    def is_category(self, node):
        try:
            wn.synset(node)
            return True
        except ValueError:
            return False
        except WordNetError:
            return False

    def num_instances(self):
        return self.num_insts

    def get_root(self):
        return 'entity.n.01'

    def get_ancestor_categories(self, node):
        if self.is_instance(node):
            synsets = wn.synsets(node)
        else:
            synsets = [wn.synset(node)]
        result = []
        for synset in synsets:
            result.append(synset.name())
            while len(synset.hypernyms()) > 0:
                synset = synset.hypernyms()[0]
                result.append(synset.name())
        return set(result)

    def get_descendant_instances(self, node):
        result = set()
        sense = wn.synset(node)
        for y in sense.hyponyms():
            for lemma in y.lemmas():
                result.add(lemma.name())
            for z in get_all_hyponyms_from_sense(y):
                for lemma in z.lemmas():
                    result.add(lemma.name())
        return result


def hypernym_chain(synset_name):
    """
    e.g. hypernym_chain('boat.n.01')

    """
    synset = wn.synset(synset_name)
    result = [synset]
    while len(synset.hypernyms()) > 0:
        synset = synset.hypernyms()[0]
        result.append(synset)
    return result


def get_all_hypernyms_from_sense(sense):
    result = set()
    for y in sense.hypernyms():
        result.add(y)
        for z in get_all_hypernyms_from_sense(y):
            result.add(z)
    return result


def get_all_hypernyms(word):
    """
    e.g. get_all_hypernyms('dog')

    """
    result = set()
    for x in wn.synsets(word):
        for y in get_all_hypernyms_from_sense(x):
            result.add(y)
    return result


def get_all_hyponyms_from_sense(sense):
    """
    e.g. get_all_hyponyms_from_sense(wn.synset('metallic_element.n.01'))

    """
    result = set()
    for y in sense.hyponyms():
        result.add(y)
        for z in get_all_hyponyms_from_sense(y):
            result.add(z)
    return result


if __name__ == "__main__":
    puzzles = list(read_ooo_puzzles_from_tsv(sys.argv[1]))
    wordnet = WordnetTaxonomy()
    similarity = TaxonomySimilarity(wordnet)
    (win, lose, tie) = solve_puzzles(puzzles, similarity)
    print("Correct: {}\nIncorrect: {}\nAbstained: {}".format(win, lose, tie))





