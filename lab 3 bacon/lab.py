"""
6.101 Lab:
Bacon Number
"""

#!/usr/bin/env python3

import pickle

# import typing # optional import
# import pprint # optional import

# NO ADDITIONAL IMPORTS ALLOWED!


def transform_data(raw_data):
    """
    Transforms raw_data to instead be a dictionary with two parts:
    1. every actor and every actor they acted with in a dict
    2. every movie and every actor in it in a dict
    """

    new_data = {}
    movies_data = {}
    for pair in raw_data:
        # actors part:
        name1 = pair[0]
        name2 = pair[1]
        if name1 in new_data:
            new_data[name1].add(name2)
        else:
            new_data[name1] = {name2}
        if name2 in new_data:
            new_data[name2].add(name1)
        else:
            new_data[name2] = {name1}

        # movies part:
        movie = pair[2]
        if movie in movies_data:
            movies_data[movie].add(name1)
            movies_data[movie].add(name2)
        else:
            movies_data[movie] = {name1, name2}

    return {"acted_with": new_data, "movies_to_actors": movies_data}
    # new_data[i[0]]
    #


def acted_together(transformed_data, actor_id_1, actor_id_2):
    """
    loop through data and check if both actors are in any one tuple
    """

    if actor_id_1 == actor_id_2:
        return True
    if actor_id_1 in transformed_data["acted_with"]:
        return actor_id_2 in transformed_data["acted_with"][actor_id_1]
    return False


def actors_with_bacon_number(transformed_data, n):
    """
    computes bacon number from data
    """
    agenda = {0: [4724]}
    seen = {4724}

    # one for each bacon level
    for i in range(n):
        # for each one in the previous bacon level
        for j in agenda[i]:
            # for each element in their acted_with
            for x in transformed_data["acted_with"][j]:
                if x not in seen:
                    if i + 1 in agenda:
                        agenda[i + 1].append(x)
                    else:
                        agenda[i + 1] = [x]
                    seen.add(x)
        if i + 1 not in agenda:
            # make sure to return early if there are no actors in that bacon level
            return set()
    if n not in agenda:
        return set()
    return set(agenda[n])


def trace_path(parents, start_element, end_element):
    """
    rebuilds the path from start_element to end_element using the parents dictionary.

    Args:
        parents: Dictionary mapping each element to its parent element in the path
        start_element: The starting element of the path
        end_element: The ending element of the path

    returns a list representing the path from start_element to end_element
    """
    current_element = parents[end_element]
    path = [end_element]
    while current_element != start_element:
        path.append(current_element)
        current_element = parents[current_element]
    path.append(start_element)
    path.reverse()
    return path




def bacon_path(transformed_data, actor_id):
    """
    uses future actor_to_actor_path function but with Bacon's number
    """
    return actor_to_actor_path(transformed_data, 4724, actor_id)


def actor_to_actor_path(transformed_data, actor_id_1, actor_id_2):
    """
    Finds the shortest path of actor ids connecting actor_id_1 to actor_id_2.

    Returns a list with the shortest path of actor ids from actor_id_1 to actor_id_2,
    or None if no path exists.
    """
    def test(actor):
        return actor == actor_id_2
    return actor_path(transformed_data, actor_id_1, test)


def movie_path(transformed_data, actor_id_1, actor_id_2):
    """
    Find a path of movies connecting two actors.

    Args:
        transformed_data: The transformed data containing actor relationships and movies
        actor_id_1: ID of the first actor
        actor_id_2: ID of the second actor

    Returns:
        A list of movie IDs that connect the actors, or None if no path exists
    """
    path = actor_to_actor_path(transformed_data, actor_id_1, actor_id_2)
    if not path:  # If no path exists, return None
        return None
    movie_path = []
    for first, second in zip(path, path[1:]):
        for key, value in transformed_data["movies_to_actors"].items():
            if first in value and second in value:
                movie_path.append(key)
                break
    return movie_path


def actor_path(transformed_data, actor_id_1, goal_test_function):
    """
    Finds the shortest path from actor_id_1
    to any actor satisfying the goal test function.

    Returns a list representing the shortest path of actor ids from actor_id_1 to the
        nearest actor satisfying the goal test, or None if no path exists.
    """
    # Checking if the starting actor satisfies the goal test
    if goal_test_function(actor_id_1):
        return [actor_id_1]
    agenda = {0: [actor_id_1]}
    parents = {actor_id_1: None}
    seen = {actor_id_1}

    # one for each bacon level
    i = 0
    while i in agenda and agenda[i]:
        # for each one in the previous bacon level
        for current_actor in agenda[i]:
            # for each element in their acted_with
            for neighbor in transformed_data["acted_with"][current_actor]:
                if neighbor not in seen:
                    # If neighbor satisfies goal test, reconstruct path and return
                    if goal_test_function(neighbor):
                        # Reconstruct path
                        path = [neighbor]
                        parent = current_actor
                        while parent is not None:
                            path.append(parent)
                            parent = parents[parent]
                        return path[::-1]  # Reverse to get start-goal order

                    # Add to next level of agenda
                    if i + 1 not in agenda:
                        agenda[i + 1] = []
                    agenda[i + 1].append(neighbor)

                    # Update parents and seen
                    parents[neighbor] = current_actor
                    seen.add(neighbor)
        i += 1

    return None


def actors_connecting_films(transformed_data, film1, film2):
    """
    Finds the shortest path of actors connecting two movies.

    Returns a list of actor ids representing the shortest path
    connecting the two movies, or None if no connection exists.
    """
    # Check if both films exist in the database
    if (
        film1 not in transformed_data["movies_to_actors"]
        or film2 not in transformed_data["movies_to_actors"]
    ):
        return None

    # Get all actors from the first film
    actors_in_film_1 = transformed_data["movies_to_actors"][film1]

    # Get all actors from the second film
    actors_in_film_2 = transformed_data["movies_to_actors"][film2]

    # Define a goal test function that checks if an actor is in film_id_2
    def is_in_film_2(actor_id):
        return actor_id in actors_in_film_2

    # Try finding a path from each actor in film_id_1
    shortest_path = None

    for start_actor in actors_in_film_1:
        path = actor_path(transformed_data, start_actor, is_in_film_2)
        if path is not None:
            # If we found a path and it's shorter than the current shortest, update
            if shortest_path is None or len(path) < len(shortest_path):
                shortest_path = path

    return shortest_path


if __name__ == "__main__":
    with open("resources/small.pickle", "rb") as f:
        smalldb = pickle.load(f)
    with open("resources/tiny.pickle", "rb") as f:
        tinydb = pickle.load(f)
    with open("resources/large.pickle", "rb") as f:
        largedb = pickle.load(f)

    with open("resources/movies.pickle", "rb") as f:
        movies = pickle.load(f)
    with open("resources/names.pickle", "rb") as f:
        names = pickle.load(f)
        # print(type(names))
        # print(names)
        print(names["Gabriela Ruffo"])
        # print(next((k for k, v in names.items() if v == 1018864)))
    data = transform_data(tinydb)
    print(data)

    # # finding movie path
    # result = movie_path(data, names["Joey Hazinsky"], names["Anton Radacic"])
    # print(result)
    # print([next((k for k, v in movies.items() if v == i)) for i in result])

    # actor to actor path
    # print(data)
    # result = actor_to_actor_path(data, 1640, 1532)
    # result = actor_to_actor_path(data, names["Gabriela Ruffo"], names["Matt Dillon"])
    # print(result)
    # print([next((k for k, v in names.items() if v == i)) for i in result])

    # 236239 = Julian Soler
    # path of actors from Kevin Bacon to Julian Soler
    # print(bacon_path(data, names["Julian Soler"]))
    # result = bacon_path(data, names["Julian Soler"])
    # print(result)
    # print([next((k for k, v in names.items() if v == i)) for i in result])

    # get names from ids step 5
    # actors with bacon number 6:
    # result = actors_with_bacon_number(data, 6)
    # print(result)
    # print([next((k for k, v in names.items() if v == i)) for i in result])
    # print(acted_together(data, 2876, 4724))
