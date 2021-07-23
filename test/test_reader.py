import unittest
from oddoneout.reader import O3Reader


class TestReader(unittest.TestCase):

    def test_reader1(self):
        reader = O3Reader()
        result = reader._read('data/anomia/common1.tsv')
        for r in result:
            print(r)
        #assert result == (0.25, 'color')




