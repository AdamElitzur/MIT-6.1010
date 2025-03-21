"""
6.101 Lab:
Recipes
"""

import pickle
import sys

# import typing # optional import
# import pprint # optional import

sys.setrecursionlimit(20_000)
# NO ADDITIONAL IMPORTS!


def atomic_ingredient_costs(recipes_db):
    """
    Given a recipes database, a list containing compound and atomic food tuples,
    make and return a dictionary mapping each atomic food name to its cost.
    """
    # new dictionary
    final = {i: j for type, i, j in recipes_db if type == "atomic"}
    return final


def compound_ingredient_possibilities(recipes_db):
    """
    Given a recipes database, a list containing compound and atomic food tuples,
    make and return a dictionary that maps each compound food name to a
    list of all the ingredient lists associated with that name.
    """
    final = {}
    for food_type, i, j in recipes_db:
        if food_type == "compound":
            if i not in final:
                final[i] = [j]
            else:
                final[i].append(j)
    return final


def lowest_cost(recipes_db, first_food_name, exclusion=None):
    """
    Given a recipes database and the name of a food (str), return the lowest
    cost of a full recipe for the given food item or None if there is no way
    to make the food_item.
    """
    atomic = atomic_ingredient_costs(recipes_db)
    compounds = compound_ingredient_possibilities(recipes_db)
    # handles the exclusion ingredients if there are any:
    if exclusion:
        for element in exclusion:
            if element in atomic:
                del atomic[element]
            elif element in compounds:
                del compounds[element]

    def recursive_lowest_cost(food):
        # base case:
        if food in atomic:
            return atomic[food]  # price
        if food not in compounds:
            return None
        mins = []
        # for each recipe
        for recipe in compounds[food]:
            curr = 0
            # for each compound in the recipe
            for compound in recipe:
                lowest_cost_call = recursive_lowest_cost(compound[0])
                if lowest_cost_call is None:
                    curr += float(
                        "inf"
                    )  # adding infinity to the curr so it won't be the min
                else:
                    # add the quantity * the lowest cost of the compound
                    curr += compound[1] * lowest_cost_call
            # after you got the lowest price of each compound in the recipe, add to mins
            mins.append(curr)

        # return the minimum from the mins list
        return min(mins) if min(mins) != float("inf") else None

        # example format:
        # atomic: {'milking stool': 5, 'cutting-edge laboratory': 1000,
        # 'time': 10000, 'cow': 100}

        # compound: {'milk': [[('cow', 1), ('milking stool', 1)]],
        # 'cheese': [[('milk', 1), ('time', 1)], [('cutting-edge laboratory', 11)]]}

    return recursive_lowest_cost(first_food_name)


def scaled_recipe(recipe_dict, n):
    """
    Given a dictionary of ingredients mapped to quantities needed, returns a
    new dictionary with the quantities scaled by n.
    """
    return {i: j * n for i, j in recipe_dict.items()}


def add_recipes(recipe_dicts):
    """
    Given a list of recipe dictionaries that map food items to quantities,
    return a new dictionary that maps each ingredient name
    to the sum of its quantities across the given recipe dictionaries.

    For example,
        add_recipes([{'milk':1, 'chocolate':1}, {'sugar':1, 'milk':2}])
    should return:
        {'milk':3, 'chocolate': 1, 'sugar': 1}
    """
    new = {}
    # print(recipe_dicts)
    for recipe in recipe_dicts:
        # print(recipe)
        for ingredient, amount in recipe.items():
            if ingredient not in new:
                new[ingredient] = amount
            else:
                new[ingredient] += amount
    return new


def cheapest_flat_recipe(recipes_db, first_food_name, exclusion=None):
    """
    Given a recipes database and the name of a food (str), return a dictionary
    (mapping atomic food items to quantities) representing the cheapest full
    recipe for the given food item.

    Returns None if there is no possible recipe.
    """
    atomic = atomic_ingredient_costs(recipes_db)
    compounds = compound_ingredient_possibilities(recipes_db)
    # handles the exclusion ingredients if there are any:
    if exclusion:
        for element in exclusion:
            if element in atomic:
                del atomic[element]
            elif element in compounds:
                del compounds[element]

    def recursive_best_recipe(food):
        # base case:
        if food in atomic:
            # ({food: quantity}, price)
            return ({food: 1}, atomic[food])
        if food not in compounds:
            return (None, float("inf"))

        min_cost = float("inf")
        best_ingredients = {}

        for recipe in compounds[food]:
            curr = 0
            curr_ingredients = {}
            # for each compound in the recipe
            for compound in recipe:
                ingredient_ingredients, ingredient_cost = recursive_best_recipe(
                    compound[0]
                )
                if not ingredient_ingredients:  # change
                    curr += float(
                        "inf"
                    )  # adding infinity to the curr so it won't be the min
                else:

                    current_scaled_recipe = scaled_recipe(
                        ingredient_ingredients, compound[1]
                    )
                    # add the quantity * the lowest cost of the compound
                    curr += compound[1] * ingredient_cost
                    curr_ingredients = add_recipes(
                        [curr_ingredients, current_scaled_recipe]
                    )  # wrong

            # after you got the lowest price of each compound in the recipe, add to mins
            if curr < min_cost:
                min_cost = curr
                best_ingredients = curr_ingredients

        # return the minimum from the mins list
        return (
            (best_ingredients, min_cost)
            if min_cost != float("inf")
            else (None, float("inf"))
        )

        # example format:
        # atomic: {'milking stool': 5, 'cutting-edge laboratory': 1000,
        # 'time': 10000, 'cow': 100}

        # compound: {'milk': [[('cow', 1), ('milking stool', 1)]],
        # 'cheese': [[('milk', 1), ('time', 1)], [('cutting-edge laboratory', 11)]]}

    return recursive_best_recipe(first_food_name)[0]


def combine_recipes(nested_recipes):
    """
    Given a list of lists of recipe dictionaries, where each inner list
    represents all the recipes for a certain ingredient, compute and return a
    list of recipe dictionaries that represent all the possible combinations of
    ingredient recipes.
    """
    # print(nested_recipes)
    if nested_recipes == []:
        return []
    if [] in nested_recipes:
        return []
    # first element combined with everything
    current = []
    rest_of_recipes_combined = combine_recipes(nested_recipes[1:])
    # print(rest_of_recipes_combined)
    for recipe in nested_recipes[0]:
        if rest_of_recipes_combined:
            for i in rest_of_recipes_combined:
                current.append(add_recipes([recipe] + [i]))
        else:
            current.append(add_recipes([recipe] + rest_of_recipes_combined))
    return current


def all_flat_recipes(recipes_db, food_name, exclusion=None):
    """
    Given a recipes database, the name of a food (str), produce a list (in any
    order) of all possible flat recipe dictionaries for that category.

    Returns an empty list if there are no possible recipes
    """
    atomic = atomic_ingredient_costs(recipes_db)
    compounds = compound_ingredient_possibilities(recipes_db)
    # handles the exclusion ingredients if there are any:
    if exclusion:
        for element in exclusion:
            if element in atomic:
                del atomic[element]
            elif element in compounds:
                del compounds[element]

    def recursive_all_recipes(food):
        # base case:
        if food in atomic:
            # ({food: quantity})
            return [{food: 1}]
        if food not in compounds:
            return []

        ingredients = []

        for recipe in compounds[food]:

            expansions = []
            for compound in recipe:
                recall = recursive_all_recipes(compound[0])
                recall = [scaled_recipe(rec, compound[1]) for rec in recall]
                expansions.append(recall)
            combined = combine_recipes(expansions)
            # combine_recipes([[{peanuts}], [{strawberries, sugar}, {blue, sugar}]])
            for i in combined:
                ingredients.append(i)
        return ingredients

    return recursive_all_recipes(food_name)


if __name__ == "__main__":
    # load example recipes from section 3 of the write-up
    with open("test_recipes/example_recipes.pickle", "rb") as f:
        example_recipes_db = pickle.load(f)
    # you are free to add additional testing code here!
    # i = atomic_ingredient_costs(example_recipes_db)
    # print(sum(i.values()))
    # i = compound_ingredient_possibilities(example_recipes_db)
    # print(sum(1 for j in i.values() if len(j) > 1))

    dairy_recipes_db = [
        ("compound", "milk", [("cow", 1), ("milking stool", 1)]),
        ("compound", "cheese", [("milk", 1), ("time", 1)]),
        ("compound", "cheese", [("cutting-edge laboratory", 11)]),
        ("atomic", "milking stool", 5),
        ("atomic", "cutting-edge laboratory", 1000),
        ("atomic", "time", 10000),
        ("atomic", "cow", 100),
    ]
    # print(lowest_cost(dairy_recipes_db, 'cheese'))

    cookie_recipes_db = [
        ("compound", "cookie sandwich", [("cookie", 2), ("ice cream scoop", 3)]),
        ("compound", "cookie", [("chocolate chips", 3)]),
        ("compound", "cookie", [("sugar", 10)]),
        ("atomic", "chocolate chips", 200),
        ("atomic", "sugar", 5),
        ("compound", "ice cream scoop", [("vanilla ice cream", 1)]),
        ("compound", "ice cream scoop", [("chocolate ice cream", 1)]),
        ("atomic", "vanilla ice cream", 20),
        ("atomic", "chocolate ice cream", 30),
    ]
    # print(lowest_cost(cookie_recipes_db, 'cookie sandwich'))

    dairy_recipes_db2 = [
        ("compound", "milk", [("cow", 1), ("milking stool", 1)]),
        ("compound", "cheese", [("milk", 1), ("time", 1)]),
        ("compound", "cheese", [("cutting-edge laboratory", 11)]),
        ("atomic", "milking stool", 5),
        ("atomic", "cutting-edge laboratory", 1000),
        ("atomic", "time", 10000),
    ]
    # print(lowest_cost(dairy_recipes_db2, 'cheese'))
    # print(lowest_cost(dairy_recipes_db2, "cheese", ["cutting-edge laboratory"]))

    soup = {
        "carrots": 5,
        "celery": 3,
        "broth": 2,
        "noodles": 1,
        "chicken": 3,
        "salt": 10,
    }

    carrot_cake = {
        "carrots": 5,
        "flour": 8,
        "sugar": 10,
        "oil": 5,
        "eggs": 4,
        "salt": 3,
    }
    bread = {"flour": 10, "sugar": 3, "oil": 3, "yeast": 15, "salt": 5}
    test_recipe_dicts = [soup, carrot_cake, bread]
    # print(add_recipes(test_recipe_dicts))
    # print(cheapest_flat_recipe(dairy_recipes_db, "cheese"))

    cookie_recipes_db = [
        ("compound", "cookie sandwich", [("cookie", 2), ("ice cream scoop", 3)]),
        ("compound", "cookie", [("chocolate chips", 3)]),
        ("compound", "cookie", [("sugar", 10)]),
        ("atomic", "chocolate chips", 200),
        ("atomic", "sugar", 5),
        ("compound", "ice cream scoop", [("vanilla ice cream", 1)]),
        ("compound", "ice cream scoop", [("chocolate ice cream", 1)]),
        ("atomic", "vanilla ice cream", 20),
        ("atomic", "chocolate ice cream", 30),
    ]

    print(all_flat_recipes(cookie_recipes_db, "cookie sandwich"))
