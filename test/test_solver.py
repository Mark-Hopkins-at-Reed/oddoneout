import unittest
from oddoneout.solver import TaxonomySimilarity, is_solvable, rank_puzzle_choices
from oddoneout.solver import solve_puzzle, solve_puzzles, silent_logger
from oddoneout.taxonomy import GraphTaxonomy
from oddoneout.puzzle import OddOneOutPuzzle


class TestSolver(unittest.TestCase):

    def setUp(self):
        self.example = GraphTaxonomy(
            'entity',
            {'apple': ['fruit'],
             'lemon': ['citrus'],
             'orange': ['citrus', 'color'],
             'peach': ['fruit', 'color'],
             'red': ['color'],
             'yellow': ['color'],
             'citrus': ['fruit'],
             'fruit': ['entity'],
             'color': ['entity'],
             'entity': []}
        )

    def test_similarity1(self):
        similarity = TaxonomySimilarity(self.example)
        result = similarity('apple', ['orange', 'red', 'peach', 'yellow'])
        assert result == (0.25, 'color')

    def test_similarity2(self):
        similarity = TaxonomySimilarity(self.example)
        result = similarity('apple', ['orange', 'lemon'])
        assert result == (0.5, 'citrus')

    def test_similarity3(self):
        similarity = TaxonomySimilarity(self.example)
        result = similarity('red', ['orange', 'peach'])
        assert result == (0.25, 'fruit')

    def test_is_recognized(self):
        similarity = TaxonomySimilarity(self.example)
        assert similarity.is_recognized("red")
        assert not similarity.is_recognized("green")

    def test_is_solvable(self):
        similarity = TaxonomySimilarity(self.example)
        puzzle = OddOneOutPuzzle("apple",
                                 ['orange', 'red', 'peach', 'yellow'],
                                 "color")
        assert is_solvable(puzzle, similarity)
        puzzle = OddOneOutPuzzle("green",
                                 ['orange', 'red', 'peach', 'yellow'],
                                 "color")
        assert not is_solvable(puzzle, similarity)
        puzzle = OddOneOutPuzzle("apple",
                                 ['orange', 'green', 'peach', 'yellow'],
                                 "color")
        assert not is_solvable(puzzle, similarity)

    def test_rank_puzzle_options1(self):
        similarity = TaxonomySimilarity(self.example)
        puzzle = OddOneOutPuzzle("red",
                                 ['orange', 'lemon'],
                                 "citrus")
        expected = [(0.5, 'citrus', 'red'),
                    (0.25, 'color', 'lemon'),
                    (1.0 / 6.0, 'entity', 'orange')]
        assert rank_puzzle_choices(puzzle, similarity) == expected

    def test_rank_puzzle_options2(self):
        similarity = TaxonomySimilarity(self.example)
        puzzle = OddOneOutPuzzle("apple",
                                 ['orange', 'peach', 'red'],
                                 "color")
        expected = [(0.25, 'fruit', 'red'),
                    (0.25, 'color', 'apple'),
                    (1.0 / 6.0, 'entity', 'peach'),
                    (1.0 / 6.0, 'entity', 'orange')]
        assert set(rank_puzzle_choices(puzzle, similarity)) == set(expected)

    def test_solve_puzzle1(self):
        similarity = TaxonomySimilarity(self.example)
        puzzle = OddOneOutPuzzle("red",
                                 ['orange', 'lemon'],
                                 "citrus")
        expected = (0.5, 'citrus', 'red')
        assert solve_puzzle(puzzle, similarity) == expected

    def test_solve_puzzle2(self):
        similarity = TaxonomySimilarity(self.example)
        puzzle = OddOneOutPuzzle("apple",
                                 ['orange', 'peach', 'red'],
                                 "color")
        assert solve_puzzle(puzzle, similarity) is None

    def test_solve_puzzle3(self):
        similarity = TaxonomySimilarity(self.example)
        puzzle = OddOneOutPuzzle("red",
                                 ['orange', 'lime'],
                                 "citrus")
        assert solve_puzzle(puzzle, similarity) is None

    def test_solve_puzzles(self):
        similarity = TaxonomySimilarity(self.example)
        puzzle1 = OddOneOutPuzzle("red",
                                  ['orange', 'lemon'],
                                  "citrus")
        puzzle2 = OddOneOutPuzzle("apple",
                                  ['orange', 'peach', 'red'],
                                  "color")
        puzzle3 = OddOneOutPuzzle("red",
                                  ['orange', 'lime'],
                                  "citrus")
        puzzle4 = OddOneOutPuzzle("orange",
                                  ['red', 'lemon'],
                                  "citrus")
        puzzles = [puzzle1, puzzle2, puzzle3, puzzle4]
        result = solve_puzzles(puzzles, similarity, logger=silent_logger)
        assert result == (1, 1, 2)




