"""
wikigraph.py

Implements the WCGTaxonomy subclass of the Taxonomy class.

It is intended to access the structure of the Wikipedia Category Graph,
derived from the enwiki categorylinks and pages files,
so it can be ordered into an ontology.
"""

import copy
from taxonomy import Taxonomy, Specificity
from wiki_demo import findPagesInCategory, findPagesById
from collections import defaultdict


class WCGTaxonomy(Taxonomy):
    
    def __init__(self, categorylinks_filename, pages_filename):
        self.specificity = Specificity()
        self.catlinks = getAllCategories(categorylinks_filename)
        self.pages = getAllPages(pages_filename)
        self.num_insts = len(self.get_descendant_instances(self.get_root()))
        
    def is_instance(self, node):
        return node in self.pages and self.specificity(self, node) == 0
    
    def is_category(self, node):
        return node in self.catlinks and self.specificity(self, node) > 0
    
    def num_instances(self):
        return self.num_insts
    
    def get_root(self):
        # For now, use "'Recipes'" as a guaranteed root with many subcategories.
        # When looking at the entire enwikibooks, use "'Categories'"
        return "'Contents'"
    
    def get_ancestor_categories(self, node):
        # Given a page label, returns a set of its ancestor category labels
        pages = self.get_page_dict()
        catlinks = self.get_catlinks_dict()
        p_ids = pages[node]
        visited_pages = p_ids.copy()
        
        categories = set()
        while len(p_ids) != 0 and self.get_root() not in categories:
            
            page_id, page_namespace = p_ids.pop()
            if page_id in catlinks:
                for cat_name, page_type in catlinks[page_id]:
                    
                    cat_id = pages[cat_name][0]
                    
                    if cat_id[0] not in visited_pages:
                        categories.add(cat_name)
                        visited_pages.append(cat_id[0])
                        p_ids.append(cat_id)
        
        return categories
    
    def get_descendant_instances(self, node):
        # Given a category label, returns a set of its descendant page labels
        
        pages = self.get_page_dict()
        catlinks = self.get_catlinks_dict()
        
        if node not in catlinks:
            return set()
        
        descendants = set()
        visited_categories = [node]
        
        categories = [node]
        while len(categories) != 0:
            category_name = categories.pop()
            for page_id, page_type in catlinks[category_name]:
                if page_id in pages:
                    for page_name, page_namespace in pages[page_id]:
                        
                        if page_type == "'page'":
                            descendants.add(page_name)
                        elif page_type == "'subcat'":
                            if page_name not in visited_categories and page_name in self.catlinks:
                                categories.append(page_name)
                                visited_categories.append(page_name)
        
        return descendants
        
    def get_page_dict(self):
        return copy.deepcopy(self.pages)
    
    def get_catlinks_dict(self):
        return copy.deepcopy(self.catlinks)

def isMetaData(page_namespace):
    """
    WCG metadata have page_namespace = 1-13 or 15.
    
    Reference: https://www.mediawiki.org/wiki/Manual:Namespace
    """
    pn = int(page_namespace)
    return pn == 15 or (pn >= 1 and pn <= 13)

def getAllPages(pages_filename):
    pages_file = open(pages_filename, 'r')
    
    pages = defaultdict(list)
    for page in pages_file:
        page_id, page_namespace, page_title, page_is_redirect, page_len, page_content_model, page_lang = page.split('\t')
        if not isMetaData(page_namespace):
            pages[page_id].append( (page_title, page_namespace) )
            pages[page_title].append( (page_id, page_namespace) )
    pages_file.close()
    return pages

def getAllCategories(catlinks_filename):
    catlinks_file = open(catlinks_filename, 'r')
    
    cats = defaultdict(list)
    for category in catlinks_file:
        page_id, cat_label, page_type = category.strip('\n').split('\t')
        
        cats[cat_label].append( (page_id, page_type) )
        cats[page_id].append( (cat_label, page_type) )
    catlinks_file.close()
    return cats

def getCategoriesOfPage(root, catlinks_filename, p_id):
    catlinks_file = open(catlinks_filename, 'r')
    
    categories = set()
    for category in catlinks_file:
        page_id, cat_label, page_type = category.strip('\n').split('\t')
        if page_id == p_id:
            categories.add(cat_label)
    
    catlinks_file.close()
    return categories

