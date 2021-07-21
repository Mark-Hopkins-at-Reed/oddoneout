import unittest
from wikigraph import WCGTaxonomy
from taxonomy import lowest_common_ancestor
from solver import solve_puzzle, TaxonomySimilarity, solve_puzzles, silent_logger
from puzzle import OddOneOutPuzzle, common2_puzzles


class TestWCG(unittest.TestCase):

    def setUp(self):
        self.taxonomy = WCGTaxonomy("../enwiki-20201020-categorylinks",
                                    "../enwiki-20201020-page")

    def test_is_instance(self):
        assert self.taxonomy.is_instance("'Easy_Peach_Cobbler'")
        assert self.taxonomy.is_instance("'Pupusa'")
        assert not self.taxonomy.is_instance("'Cobbler_recipes'")
        assert not self.taxonomy.is_instance("'Asdfg_lkj'")

    def test_is_category(self):
        assert not self.taxonomy.is_category("'Afghan_Bread'")
        assert not self.taxonomy.is_category("'Chicken_Tikka'")
        assert self.taxonomy.is_category("'Sindhi_recipes'")
        assert not self.taxonomy.is_category("'Something_else'")

    def test_ancestor_categories1(self):
        result = self.taxonomy.get_ancestor_categories("'Vegan_recipes'")
        expected = ["'Recipes'", "'Recipes_by_diet'", "'Vegetarian_recipes'"]
        """
        # These additional categories will be added if the taxonomy
        # has root Categories, not root Recipes
        expected = ["'Cataloguing'", "'Categories'", "'Cookbook'", 
                    "'Hidden_categories'", "'Recipes'", "'Recipes_by_diet'", 
                    "'Vegetarian_recipes'"]
        """
        assert sorted(result) == expected

    def test_ancestor_categories2(self):
        result = self.taxonomy.get_ancestor_categories("'Spinach_Calzones'")
        expected = ["'Baked_recipes'", "'Cheese_recipes'", 
                    "'Main_course_recipes'", "'Pizza_recipes'", "'Recipes'", 
                    "'Recipes_with_metric_units'", "'Spinach_recipes'", 
                    "'Vegetarian_recipes'"]
        """
        # These additional categories will be added if the taxonomy
        # has root Categories, not root Recipes
        expected =  ["'Afghan_recipes'", "'Appetizer_recipes'", 
                     "'Cataloguing'", "'Categories'", "'Chicken_recipes'", 
                     "'Cookbook_utility_categories'", "'Easy_recipes'", 
                     "'Featured_recipes'", "'Hidden_categories'", 
                     "'Indian_recipes'", "'Pakistani_recipes'", 
                     "'Pashtun_recipes'", "'Punjabi_recipes'", "'Recipes'", 
                     "'Recipes_with_images'", "'Recipes_with_metric_units'"]
        """
        assert sorted(result) == expected

    def test_descendant_instances1(self):
        result = self.taxonomy.get_descendant_instances("'Vegan_recipes'")
        expected = ["'Ale,_Mustard_and_Winter_Vegetable_Pie'",
                    "'Almond-Cowitch_Matcha_Smoothie'", "'Almond_Milk'", 
                    "'Anarchist_Oatmeal_Cookies'", "'Appam'", 
                    "'Apple_Parsnip_Soup'", "'Aquafaba'", "'Azteca'", 
                    "'Baba_Ganoush'", "'Baguette'", "'Baked_Eggplant'", 
                    "'Bean-Semolina_Burger'", "'Bean_Chili'", 
                    "'Bean_Jahni_soup'", "'Bean_Spelt_Oat_Spread'", 
                    "'Beans_and_Rice'", "'Black-Eyed_Peas_and_Kale'", "'Bonda'",
                    "'Brazilian_Black_Bean_Soup'", 
                    "'Bread_Filled_with_Potato_Curry_(Pani_Puri)'", 
                    "'Brown_Rice_with_Other_Starches_and_Vegetables_(Pandora\\'s_Feast)'",
                    "'Bulgher_Burger'", "'Bulgur_Bread'", "'Calentita'", 
                    "'Caramel_Sauce'", "'Caramel_Sauce_(vegan)'", 
                    "'Cardamom_Bread'", "'Carrot_Paprika_Spread'", 
                    "'Carrot_Salad'", "'Carrot_and_Raisin_Salad'", 
                    "'Carrot_and_coriander_soup'", 
                    "'Carrots_Cooked_Kinpira_Style'", "'Cashew_Gravy'", 
                    "'Chapati'", "'Charoset'", "'Charoset_Maghrebi'", 
                    "'Cheddar_(Vegan)'", "'Chickpea_Curry_(Masaledaar_chole)'",
                    "'Chickpea_Stew'", "'Chili_(Vegan)'", "'Chocolate_Balls'", 
                    "'Chocolate_Chip_Cookies_(Vegan)'",
                    "'Chocolate_Mousse_(Vegan)'",
                    "'Chocolate_Mousse_Cake_(Vegan)'", "'Cholley'", 
                    "'Chow_Mein'", "'Ciceri_e_Tria'", "'Coconut_Chutney'", 
                    "'Coconut_Chutney_(North_Indian)'", 
                    "'Corn_Bread_(Lombardy)'", "'Cornbread'", 
                    "'Creamy_Aioli_Salsa'", 
                    "'Curried_Chiles_(Mirchi_ka_Salan)'", 
                    "'Deep_Fried_Chickpea_Dough_Balls_(Chyueeam)'", 
                    "'Deep_Fried_Chickpea_Dough_Curry_Snacks_(Pakoda)'", 
                    "'Deep_Fried_Chickpea_Dough_with_Lemon_(Panissa)'", 
                    "'Dosa'", "'Double_Pea_Soup'", "'E\\xc3\\xa4rgon_Salsa'", 
                    "'Eggplant_Pasta'", "'Eggplant_and_Chickpea_Skillet'", 
                    "'Eggplant_and_Tahini_(Moutabbal)'", 
                    "'English_Field_Bean_Pate'", "'Falafel_Seitan_Bratwurst'", 
                    "'Falafel_salad'", "'Fish_Sauce_(Vegan)'", 
                    "'Fresh_Broadbeans_Salad'", "'Fried_Rice'", "'Fry_Bread'", 
                    "'Gallo_pinto'", "'Garlic_Croutons'", "'Gazpacho'", 
                    "'Ghavoot'", "'Gluten-Free_Blackberry_Oat_Bars'", 
                    "'Gluten-Free_Quinoa_Vegetable_Slaw_With_Peanut_Dressing'", 
                    "'Gluten_Veggie_Meat'", "'Grape_leaves'", 
                    "'Green_Leaf_Detoxifier'", 
                    "'Green_Mango_and_Cumin_Drink_(Aam_Panna)'", 
                    "'Grilled_Aubergine'", "'Grilled_Portobello_Mushrooms'", 
                    "'Grilled_Vegetable_Sandwich'", "'Guacamole'", 
                    "'Hard_Tack'", "'Harrisa'", "'Hilbah'", 
                    "'Homemade_Pie_Crust'", "'Hummus_(Greek)'", 
                    "'Hummus_I'", "'Iced_Tea'", "'Idli'", "'Indian_Beans'", 
                    "'Indian_Potatoes'", "'Indian_Rice'", "'Italian_Bread'", 
                    "'Keralan_vegetable_stew'", "'Khaman'", "'Khus_Khus_Halwa'", 
                    "'Koshary'", "'Kyopulo'", "'Lemon_Pickle'", 
                    "'Lemon_Pickle_2'", "'Lentil_Soup'", 
                    "'Lentils_and_Rice_(Mjeddrah)'", 
                    "'Macaroni_&_Nutritional_Yeast'", "'Mango_Avocado_Salsa'", 
                    "'Mango_Chutney'", "'Matzo'", "'Meatloaf_V'", "'Medu_vada'", 
                    "'Mike\\'s_Bean_and_Rice_Bake'", 
                    "'Mike\\'s_Saffron_Rice_and_Beans'", "'Minestra'", 
                    "'Mixed_vegetables'", "'Montreal_Russian_Borscht'", 
                    "'Mung_Bean_and_Brown_Rice_Curry'", 
                    "'No-Bake_Chocolate_Pie'", "'Oat_Milk'", 
                    "'Olive_Oil_Bread'", "'Onion_Chutney'", 
                    "'Oven-Roasted_Potatoes'", "'Oven_Pancakes_(Vegan)'", 
                    "'Pain_au_Levain_Naturel'", "'Pan-Fried_Shanghai_Noodles'", 
                    "'Pancake'", "'Pancakes_(Vegan)'", "'Pasta_Soup_for_Kids'", 
                    "'Pasta_and_Bean_Soup_(Pasta_e_Fagioli)'", 
                    "'Peach_and_Tomato_Gazpacho'", 
                    "'Peanut-Butter_Banana_Smoothie'", "'Peanut_Milk'", 
                    "'Peanut_Sauce_(Vegan)'", "'Pear_chutney'", 
                    "'Pecan_Spinach_Pasta'", "'Pita'", "'Pohe'", "'Polenta'", 
                    "'Potato-Chickpea_Curry'", "'Potato_Curry_(Aloo_Masala)'", 
                    "'Potato_Samosas_(Aloo_Pies)'", "'Potato_Spelt_Oat_Burger'", 
                    "'Preserved_Lemon'", "'Puliyodarai_Quick'", 
                    "'Pulse_Chutney'", "'Puttanesca_Sauce_(Vegan)'", "'Qabuli'", 
                    "'Quinoa,_Shiitake_Mushrooms_and_Adzuki_Beans'", 
                    "'Ragi_Dosa'", "'Raspberry_Shortbread_Bars'", 
                    "'Ratatouille'", "'Ratatouille_2'", "'Rhubarb_Pie'", 
                    "'Rice-Oat-Semolina_Burger'", "'Rice_Wheat_Spread'", 
                    "'Rice_an\\'_Peas'", "'Rice_and_Lentils_(Mejadra)'", 
                    "'Rice_with_Black_Beans_(Arroz_con_Frijoles_Negros)'", 
                    "'Rice_with_Lemon_Coconut_and_Eggplant_(Vangibhat)'", 
                    "'Rice_with_Tofu_and_Nuts'", "'Roasted_Eggplant'", 
                    "'Rosemary_Garlic_Baked_Potatoes'", "'Rustic_Beer_Bread'", 
                    "'Rye_Bread'", "'Salsa'", "'Salsa_(Fermented)'", "'Sambar'", 
                    "'Sambar_Variation'", "'Samosa'", "'Sauerkraut'", 
                    "'Seaweed_Salad'", "'Seitan'", "'Seitan_Bratwurst'", 
                    "'Seitan_Burger'", "'Semolina_Burger'", "'Sesame_Milk'", 
                    "'Simmering_Tofu_Stir-fry'", 
                    "'Simple_Ayurvedic_Sprout_Salad'", 
                    "'Smokey_Millet_Super_Grain_Bowl_With_Crispy_Tofu'", 
                    "'Southwest_Pasta'", "'Southwestern_Black_Beans_and_Rice'", 
                    "'Soy_Milk'", "'Spelt_Burger'", "'Spicy_Carrot_Aioli'", 
                    "'Spicy_Hot_Salsa'", "'Spring_Onion_Spelt_Spread'", 
                    "'Strawberry_Rhubarb_Pie'", "'Stuffed_Kabocha_Squash'", 
                    "'Sushi/Stewed_Vegan_Maki'", "'Sushi/Sushi_Rice'", 
                    "'Sweet_Rice_II'", "'Tabouli'", "'Tabula'", "'Tadka_Dhal'", 
                    "'Tandoori_Tofu'", "'Tapai'", "'Tembleque'", 
                    "'Tempe_Mendoan'", "'Tempeh'", "'Three_Ginger_Soup'", 
                    "'Three_Sisters_Stew'", "'Tofu_Bacon_(Vegan)'", 
                    "'Tofu_Pancake'", "'Tomato_Juice'", "'Tomato_Salad'", 
                    "'Tost\\xc3\\xb3n'", "'Traditional_Pilau_Rice'", "'Ugali'", 
                    "'Undhiyu'", "'Upeseru'", "'Vanilla_Caramel'", 
                    "'Vegan_Cuisine'", "'Vegan_Eggnog_Milkshake'", 
                    "'Vegan_Green_Bean_Casserole'", "'Vegan_Lemon_Meringue_Pie'", 
                    "'Vegan_Mayonnaise'", "'Vegan_Paprika_Bratwurst'", 
                    "'Vegan_Pate'", "'Vegan_Substitutions'", 
                    "'Vegetable_Stew_and_Dumplings'", "'Virgin_Vegan_Eggnog'", 
                    "'Vitumbua'", 
                    "'Warm_Black_Bean_Salad_with_Kale_and_Tomatoes'", 
                    "'Wheat_Grass_Juice'", "'White_Bean_Soup_(Bob_Chorba)'", 
                    "'Whole_Wheat_Bread_(Vegan)'", 
                    "'Whole_Wheat_Pancakes_(Vegan)'", 
                    "'Wild_Spinach_and_Lemon_Salad'", "'Yogurt_Cupcake'", 
                    "'Yorkshire_Pudding_(Vegan)'"]
        assert sorted(result) == expected

    def test_descendant_instances2(self):
        result = self.taxonomy.get_descendant_instances("'Thanksgiving_recipes'")
        expected = ["'Candied_Sweet_Potatoes'", "'Candied_Yams'", 
                    "'Delicious_Pumpkin_Pie'", "'Holiday_Stuffing'", 
                    "'One-Hour_Thanksgiving_Dinner'", "'Pumpkin_Pie'", 
                    "'Roasted_Brined_Turkey'", "'Turkey_Biscuit'", 
                    "'Whole,_Roasted_Turkey_with_Stuffing'", "'Yam_Casserole'"]
        assert sorted(result) == expected

    def test_lowest_common_ancestor1(self):
        result = lowest_common_ancestor(self.taxonomy,
                                        ["'Chow_Mein'", "'Ghavoot'",
                                         "'Ragi_Dosa'", "'Vegan_Pate'"],
                                        "'Roasted_Brined_Turkey'")
        assert result == (229, "'Vegan_recipes'")

    def test_lowest_common_ancestor2(self):
        result = lowest_common_ancestor(self.taxonomy,
                                        ["'Chow_Mein'", "'Ghavoot'",
                                         "'Ragi_Dosa'", "'Vegan_Pate'"],
                                        "'Soy_Milk'")
        assert result == (3122, "'Recipes'")

    def test_lowest_common_ancestor3(self):
        result = lowest_common_ancestor(self.taxonomy,
                                        ["'Sausage_and_Tortellini_Soup'",
                                         "'Zuppa_alla_modenese'",
                                         "'Spinach_Calzones'", 
                                         "'Albanian_Vegetable_Pie'"],
                                        "'Rye_Bread'")
        assert result == (20, "'Spinach_recipes'")

    def test_solve_puzzle(self):
        similarity = TaxonomySimilarity(self.taxonomy)
        puzzle = OddOneOutPuzzle("'Turkey_Wrap'",
                                 ["'Candied_Sweet_Potatoes'", 
                                  "'Holiday_Stuffing'",
                                  "'Whole,_Roasted_Turkey_with_Stuffing'",
                                  "'Yam_Casserole'"],
                                 "'Thanksgiving_recipes'")
        result = solve_puzzle(puzzle, similarity)
        assert result == (0.1, "'Thanksgiving_recipes'", "'Turkey_Wrap'")

    def test_solve_puzzles(self):
        similarity = TaxonomySimilarity(self.taxonomy)
        result = solve_puzzles(common2_puzzles, similarity, logger=silent_logger)
        assert result == (38, 14, 50)

