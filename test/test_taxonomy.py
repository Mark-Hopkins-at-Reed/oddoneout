import unittest
from oddoneout.taxonomy import GraphTaxonomy, lowest_common_ancestor


class TestTaxonomy(unittest.TestCase):

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

    def test_num_instances(self):
        assert self.example.num_instances() == 6

    def test_is_instance(self):
        assert self.example.is_instance('apple')
        assert self.example.is_instance('lemon')
        assert self.example.is_instance('orange')
        assert self.example.is_instance('peach')
        assert self.example.is_instance('red')
        assert self.example.is_instance('yellow')
        assert not self.example.is_instance('citrus')
        assert not self.example.is_instance('fruit')
        assert not self.example.is_instance('color')
        assert not self.example.is_instance('entity')

    def test_is_category(self):
        assert not self.example.is_category('apple')
        assert not self.example.is_category('lemon')
        assert not self.example.is_category('orange')
        assert not self.example.is_category('peach')
        assert not self.example.is_category('red')
        assert not self.example.is_category('yellow')
        assert self.example.is_category('citrus')
        assert self.example.is_category('fruit')
        assert self.example.is_category('color')
        assert self.example.is_category('entity')

    def test_get_descendant_instances(self):
        result = self.example.get_descendant_instances('citrus')
        assert result == {'lemon', 'orange'}
        result = self.example.get_descendant_instances('color')
        assert result == {'orange', 'peach', 'red', 'yellow'}
        result = self.example.get_descendant_instances('fruit')
        assert result == {'lemon', 'orange', 'apple', 'peach'}

    def test_get_ancestor_categories(self):
        result = self.example.get_ancestor_categories('lemon')
        assert result == {'citrus', 'fruit', 'entity'}
        result = self.example.get_ancestor_categories('peach')
        assert result == {'color', 'fruit', 'entity'}
        result = self.example.get_ancestor_categories('red')
        assert result == {'color', 'entity'}
        result = self.example.get_ancestor_categories('orange')
        assert result == {'citrus', 'fruit', 'color', 'entity'}
        result = self.example.get_ancestor_categories('citrus')
        assert result == {'citrus', 'fruit', 'entity'}

    def test_lowest_common_ancestor1(self):
        result = lowest_common_ancestor(self.example,
                                        ['orange', 'lemon'],
                                        'apple')
        assert result == (2, 'citrus')

    def test_lowest_common_ancestor2(self):
        result = lowest_common_ancestor(self.example,
                                        ['orange', 'lemon', 'peach'],
                                        'apple')
        assert result == (6, 'entity')

    def test_lowest_common_ancestor3(self):
        result = lowest_common_ancestor(self.example,
                                        ['orange', 'peach'],
                                        'lemon')
        assert result == (4, 'color')

    def test_lowest_common_ancestor4(self):
        result = lowest_common_ancestor(self.example,
                                        ['orange', 'peach'],
                                        'red')
        assert result == (4, 'fruit')
