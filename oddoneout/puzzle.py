import codecs
import random


class OddOneOutPuzzle:
    def __init__(self, oddone, wordset, category):
        self.oddone = oddone
        self.wordset = wordset
        self.category = category

    def get_choices(self):
        return [self.oddone] + self.wordset

    def __str__(self):
        return str([self.oddone] + self.wordset)


def read_ooo_puzzles_from_tsv(filename):
    with codecs.open(filename, 'r', encoding='UTF-8') as reader:
        for line in reader:
            fields = line.split('\t')
            yield OddOneOutPuzzle(fields[1],
                                  [x.strip() for x in fields[2:]],
                                  fields[0])


common2_puzzles = list(read_ooo_puzzles_from_tsv("data/anomia/common2.tsv"))


def read_category_map_from_csv(csv_filename):
    """
    e.g. read_category_map_from_csv("data/colors.csv")

    """
    category_map = dict()
    with open(csv_filename) as inhandle:
        for line in inhandle:
            fields = [example for example in line.strip().split(',') if example != '']
            category_map[fields[0]] = fields[1:]
    return category_map


def generate_puzzle(category_map, category):
    """
    Given a dictionary that maps categories to examples, generates
    an odd-man-out puzzle whose theme is the given category.

    """

    examples = category_map[category]
    random.shuffle(examples)
    categories = set([str(k) for k in category_map.keys()])
    categories.remove(str(category))
    another_category = random.choice(list(categories))
    another_category_examples = category_map[another_category]
    random.shuffle(another_category_examples)
    oddman = another_category_examples[0]
    puzzle = [str(category), str(oddman)] + examples[:4]
    return puzzle


def generate_puzzles(category_map, num_puzzles_per_category):
    """
    Generates several odd-man-out puzzles.

    """
    puzzles = []
    for i in range(num_puzzles_per_category):
        for category in category_map:
            puzzle = generate_puzzle(category_map, category)
            puzzles.append(puzzle)
    return puzzles
