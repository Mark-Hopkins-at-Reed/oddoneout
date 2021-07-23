"""
ooo_reader.py

Reads in odd man out puzzles from the odd-man-out repo from
gabrielStanovsky's github, assuming the repo is in
the current relative directory.

Each puzzle is stored as a list with the same format:
    [category, odd one out, member1, member2, member3, member4]
Each list is stored in one larger list, which is returned from main.
"""

import random
from os import listdir
from os.path import isfile, join

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def read_stanovsky(puzzlefile_name):
    with open(puzzlefile_name, 'r') as reader:
        for line in reader:
            puzzle = [f.strip() for f in line.split('\t')]
            category, oddone, others = puzzle[0], puzzle[1], puzzle[2:]
            yield category, oddone, others


def write_o3(puzzles, outfile, num_shuffles=5):
    with open(outfile, 'w') as writer:
        for (_, oddone, others) in puzzles:
            for _ in range(num_shuffles):
                oddone_index = random.randint(0, 4)
                shuffled = [o for o in others]
                random.shuffle(shuffled)
                choices = shuffled[:oddone_index] + [oddone] + shuffled[oddone_index:]
                choice_str = ALPHABET[oddone_index] + "\t" + "\t".join(choices)
                writer.write(choice_str + "\n")


def rewrite_all(data_dir, output_dir):
    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    for file in onlyfiles:
        try:
            puzzles = read_stanovsky(join(data_dir, file))
            write_o3(puzzles, join(output_dir, file))
        except ValueError:
            print("Failed to convert {}".format(file))


if __name__ == "__main__":
    rewrite_all('data/anomia', 'data/anomia/shuffled')
