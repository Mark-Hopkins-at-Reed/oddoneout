"""
ooo_reader.py

Reads in odd man out puzzles from the odd-man-out repo from
gabrielStanovsky's github, assuming the repo is in
the current relative directory.

Each puzzle is stored as a list with the same format:
    [category, odd one out, member1, member2, member3, member4]
Each list is stored in one larger list, which is returned from main.
"""

import re # re.split - swifter parsing of each puzzle

def main(puzzlefile_name):
    puzzlefile = open(puzzlefile_name, 'r')
    
    dataset = []
    for puzzle in puzzlefile:
        puzzle_list = [item for item in filter(None, re.split('[\t,\n]', puzzle))]
        dataset.append(puzzle_list)
        
        # Potentially relevant data subsets
        category = puzzle_list[0]
        words = puzzle_list[1:]
        odd_one = puzzle_list[1]
        members = puzzle_list[2:]
    
    puzzlefile.close()
    return dataset

if __name__ == "__main__":
    # Current relative directory
    DATA_DIR = "odd-man-out/data/"

    # .tsv data files
    COMMON1 = DATA_DIR + "common1.tsv"
    COMMON2 = DATA_DIR + "common2.tsv"
    PROPER1 = DATA_DIR + "proper1.tsv"
    PROPER2 = DATA_DIR + "proper2.tsv"
    CROWD_FILTER = DATA_DIR + "crowdsourced_filtered.tsv"
    # unfiltered does not have typical OOO format and will not be parsed correctly
    CROWD_UNFILTER = DATA_DIR + "crowdsourced_unfiltered.tsv"
    
    main(COMMON1)