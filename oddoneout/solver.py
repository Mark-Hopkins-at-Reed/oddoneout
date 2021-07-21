from nltk.corpus import wordnet as wn
from oddoneout.taxonomy import lowest_common_ancestor


def verbose_logger(s):
    print(s)


def silent_logger(s):
    pass


def solve_puzzle(puzzle, similarity):
    choices = puzzle.wordset + [puzzle.oddone]
    scores = similarity(choices)
    if scores is None:
        return None
    result = sorted(zip(scores, choices))
    ranks = list(reversed(result))
    if ranks[0][0] == ranks[1][0]:
        return None
    else:
        return ranks[0]


def solve_puzzles(puzzles, model, logger=silent_logger):
    correct = 0
    incorrect = 0
    unattempted = 0
    for puzzle in puzzles:
        solution = solve_puzzle(puzzle, model)
        if solution is None:
            logger('*ABSTAIN*: {}'.format(puzzle))
            unattempted += 1
        else:
            (score, hypothesis) = solution
            if hypothesis == puzzle.oddone:
                logger('*CORRECT* {}: {}'.format(hypothesis, puzzle))
                correct += 1
            else:
                logger('*INCORRECT* {}: {}'.format(hypothesis, puzzle))
                incorrect += 1
                logger(' '.join(puzzle.wordset + [puzzle.oddone]))
                logger("Incorrect: " + str(hypothesis) + " should be " + str(puzzle.oddone))
    return correct, incorrect, unattempted




