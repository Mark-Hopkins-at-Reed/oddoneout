"""
wiki_demo.py

This code accesses the page and categorylinks files extracted from
Wikipedia dumps (download and format using download.py) and
puts the titles of all pages linked to by a certain category into a list,
then returns the list.
"""

def findPagesInCategory(catlinks_filename, desired_category):
    catlinks_file = open(catlinks_filename, 'r')
    
    subcat_ids = []
    file_ids= []
    page_ids = []
    
    # Find instances of desired category in categorylinks file
    for category in catlinks_file:
        page_id, cat_label, page_type = category.strip('\n').split('\t')
        
        if cat_label == desired_category:
            if page_type == "'page'":
                page_ids.append(page_id)
            elif page_type == "'subcat'":
                subcat_ids.append(page_id)
            elif page_type == "'file'":
                file_ids.append(page_id)
    
    catlinks_file.close()
    return page_ids, subcat_ids, file_ids

def findPagesById(page_filename, id_list):
    page_file = open(page_filename, 'r')
    pages_in_category = dict()
    
    for page in page_file:
        page_id, page_namespace, page_title, page_is_redirect, page_len, page_content_model, page_lang = page.split('\t')
        
        if page_id in id_list:
            pages_in_category[page_title] = [page_id, page_namespace]
            id_list.remove(page_id)
    
    page_file.close()
    return pages_in_category

def main():
    
    # Initialize variables for current search
    my_dump_date = "20201020"
        
    # To test with enwiki, finding all Dog Breed pages:
    # (https://en.wikipedia.org/wiki/Category:Dog_breeds)
    # wiki = "enwiki"
    # desired_category = "'Dog_breeds'"

    # To test with enwikibooks, smaller set with fewer subcategories:
    # (https://en.wikibooks.org/wiki/Category:Book:Organic_Chemistry)
    wiki = "enwikibooks"
    # For finding many pages:
    # desired_category = "'Organic_Chemistry'"
    # For devising a taxonomy from many subcategories:
    desired_category = "'Recipes'"
    
    catlinks_filename = wiki + "-" + my_dump_date + "-categorylinks"
    pages_filename = wiki + "-" + my_dump_date + "-page"
    
    # Collect data from txt files
    page_ids, subcat_ids, file_ids = findPagesInCategory(catlinks_filename, desired_category)
    pages = findPagesById(pages_filename, page_ids)
    subcats = findPagesById(pages_filename, subcat_ids)
    files = findPagesById(pages_filename, file_ids)
    
    # Print results for debugging and clarity
    print ("Pages in category", desired_category, ":")
    for p in pages:
        print(p)
    print("Subcategories in category", desired_category, ":")
    for s in subcats:
        print(s)
    print("Files in category", desired_category, ":")
    for f in files:
        print(f)
    
    """
    # Prints terminal children by subcategory
    while(subcats):
        sub = list(subcats.keys())[0]
        page_ids, subcat_ids, file_ids = findPagesInCategory(catlinks_filename, sub)
        pages = findPagesById(pages_filename, page_ids)
        subs = findPagesById(pages_filename, subcat_ids)
        subcats.update(subs)
        print()
        print("Kids of", sub, ":")
        for p in pages:
            print(p)
        subcats.pop(sub)
    """
    
    return 0

if __name__ == "__main__":
    main()