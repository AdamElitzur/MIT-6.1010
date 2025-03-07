"""
6.1010 Lab:
Snekoban Game
"""

# import json # optional import for loading test_levels
# import typing # optional import
# import pprint # optional import

# NO ADDITIONAL IMPORTS!


DIRECTION_VECTOR = {
    "up": (-1, 0),
    "down": (+1, 0),
    "left": (0, -1),
    "right": (0, +1),
}
SIZE = []
WALLS = set()
TARGETS = set()


def make_new_game(level_description):
    """
    Given a description of a game state, create and return a game
    representation of your choice.

    The given description is a list of lists of lists of strs, representing the
    locations of the objects on the board (as described in the lab writeup).

    For example, a valid level_description is:

    [
        [[], ['wall'], ['computer']],
        [['target', 'player'], ['computer'], ['target']],
    ]

    The exact choice of representation is up to you; but note that what you
    return will be used as input to the other functions.

    maybe
    {
    "wall": ((0,0), (0,1))
    "player": ((2,3))
    "target": ()
    "computer": ()
    }

    {'wall': {(4, 0), (5, 4), (5, 1), (0, 2), (0, 5), (1, 0),
    (2, 5), (3, 0), (4, 5), (5, 0), (5, 3), (0, 1), (0, 4), (1, 5),
      (3, 5), (5, 2), (4, 4), (5, 5), (0, 0), (0, 3), (2, 0), (1, 4), (2, 3)},
    'player': {(2, 4)},
    'target': {(1, 3)},
    'computer': {(3, 3)}}

    start with where the player is in queue, then pop it add all the next possible moves
    pop from the end, add children to the front
    """
    global SIZE

    new = {
        "wall": set(),
        "player": set(),
        "target": set(),
        "computer": set(),
    }
    SIZE = [len(level_description), len(level_description[0])]
    for row_index, row in enumerate(level_description):
        for index, value in enumerate(row):
            if len(value) > 1:
                # if there are multiple things on one index, adds all to the new dict
                for element in value:
                    new[element].add((row_index, index))
            elif len(value) == 1:
                new[value[0]].add((row_index, index))

    return (
        frozenset({list(new["player"])[0]}),
        frozenset(new["wall"]),
        frozenset(new["target"]),
        frozenset(new["computer"]),
    )
    # index key:
    # 0: player, 1: wall, 2: target, 3: computer


def neighbors(new_game):
    """
    returns the valid next moves for the user to make
    """
    valid_neighbors = []

    current = list(new_game[0])[0]  # current is the index of the player
    for direction in ["up", "down", "right", "left"]:
        new_coordinates = []

        for i, j in zip(current, DIRECTION_VECTOR[direction]):
            new_coordinates.append(i + j)
        # making sure it's not moving into a wall
        if tuple(new_coordinates) not in new_game[1]:
            # moving into an empty space or target
            if tuple(new_coordinates) not in new_game[3]:
                valid_neighbors.append((tuple(new_coordinates), direction))
                # ((1,2), "up") for example

        if tuple(new_coordinates) in new_game[3]:
            # moving into a computer:
            new_computer_coordinates = []
            # Add the direction vector to current player position
            for i, j in zip(tuple(new_coordinates), DIRECTION_VECTOR[direction]):
                new_computer_coordinates.append(i + j)
            if (
                tuple(new_computer_coordinates) not in new_game[1]
                and tuple(new_computer_coordinates) not in new_game[3]
            ):
                valid_neighbors.append((tuple(new_coordinates), direction))
    return valid_neighbors


# notes
# - no global vars???

def remove_walls_and_targets(game):
    return (game[0], game[3])

# game[1]: step_game,neighbors, game[2]: victory_check
def bfs(original_game):
    """
    finds and return shortest path from start to a node that
    satisfies victory_check, or None if no path exists, using BFS.
    """
    # start is the index of the player at the start
    start = list(original_game[0])[0]

    if victory_check(original_game):
        return (start,)

    # make an agenda. list of tuples of tuple(paths, games)
    agenda = [((start,), original_game)]
    condensed_game = remove_walls_and_targets(original_game)
    visited = {condensed_game,}  # visited stores game states instead of indices
    # visited can only have player and computers
    while agenda:
        agenda_element = agenda.pop(0)  # remove first path from agenda
        path = agenda_element[0]
        current_game = agenda_element[1]

        for neighbor in neighbors(current_game):  # looping over each possible nextmove
            # move to make a new new game for each neighbor

            new_game = step_game(current_game, neighbor[1])
            new_condensed_game = remove_walls_and_targets(new_game)

            if new_condensed_game not in visited:  # if new_game wasn't already considered
                if victory_check(new_game):
                    # return new_game if it passes goal test
                    return path + (neighbor[0],)
                agenda.append((path + (neighbor[0],), new_game))
                visited.add(new_condensed_game)
            else:
                # print("prune moves") # prune moves
                pass
    return None


def make_hashable(state):
    """
    converts a state into a hashable tuple
    """
    return (
        frozenset(state[0]),
        frozenset(state[1]),
        frozenset(state[2]),
        frozenset(state[3]),
    )


def make_mutable(state):
    """
    converts a state into a mutable tuple
    """
    return (
        set(state[0]),
        set(state[1]),
        set(state[2]),
        set(state[3]),
    )


def victory_check(game):
    """
    Given a game representation (of the form returned from make_new_game),
    return a Boolean: True if the given game satisfies the victory condition,
    and False otherwise.
    checks every computer to make sure they are in the target set
    """
    if len(game[2]) == 0:
        return False
    return all(computer in game[2] for computer in game[3])



def step_game(game, direction):
    """
    Given a game representation (of the form returned from make_new_game),
    return a game representation (of that same form), representing the
    updated game after running one step of the game.  The user's input is given
    by direction, which is one of the following:
        {'up', 'down', 'left', 'right'}.

    This function should not mutate its input.
    """
    new_game = list(make_mutable(game))
    # targets, walls always stay the same note
    new_coordinates = (
        list(new_game[0])[0][0] + DIRECTION_VECTOR[direction][0],
        list(new_game[0])[0][1] + DIRECTION_VECTOR[direction][1],
    )

    # making sure it's not moving into a wall
    if new_coordinates not in new_game[1]:
        # moving into an empty space or target
        if new_coordinates not in new_game[3]:
            new_game[0] = {new_coordinates}

    if new_coordinates in new_game[3]:
        # moving into a computer:
        new_computer_coordinates = []
        # Add the direction vector to current player position
        for i, j in zip(new_coordinates, DIRECTION_VECTOR[direction]):
            new_computer_coordinates.append(i + j)
        if (
            tuple(new_computer_coordinates) not in new_game[1]
            and tuple(new_computer_coordinates) not in new_game[3]
        ):
            new_game[3].remove(new_coordinates)
            new_game[3].add(tuple(new_computer_coordinates))
            new_game[0] = {tuple(new_coordinates)}

    return tuple(make_hashable(new_game))


def dump_game(game):
    """
    Given a game representation (of the form returned from make_new_game),
    convert it back into a level description that would be a suitable input to
    make_new_game (a list of lists of lists of strings).

    This function is used by the GUI and the tests to see what your game
    implementation has done, and it can also serve as a rudimentary way to
    print out the current state of your game for testing and debugging on your
    own.
    """

    final = [[[] for _ in range(SIZE[1])] for _ in range(SIZE[0])]
    names = ["player", "wall", "target", "computer"]

    for i, val in enumerate(game):
        for element in val:
            final[element[0]][element[1]].append(names[i])
    return final


def convert_path_to_moves(path):
    """
    takes path, converts it to moves like up or down
    """
    finals = []
    if not path:
        return None
    # goes through the whole path
    for i in range(len(path) - 1):
        # finds the change, and figures out its direction
        change = (path[i][0] - path[i + 1][0], path[i][1] - path[i + 1][1])
        if change == (0, -1):
            finals.append("right")
        elif change == (0, 1):
            finals.append("left")
        elif change == (1, 0):
            finals.append("up")
        elif change == (-1, 0):
            finals.append("down")
    return finals


def solve_puzzle(game):
    """
    Given a game representation (of the form returned from make_new_game), find
    a solution.

    Return a list of strings representing the shortest sequence of moves ("up",
    "down", "left", and "right") needed to reach the victory condition.

    If the given level cannot be solved, return None.
    """
    global WALLS
    global TARGETS
    WALLS = game[1]
    TARGETS = game[2]
    return convert_path_to_moves(bfs(game))


if __name__ == "__main__":
    test_game = [
        [["wall"], ["wall"], ["wall"], ["wall"], ["wall"], ["wall"]],
        [["wall"], [], [], ["target"], ["wall"], ["wall"]],
        [["wall"], [], [], ["wall"], ["player"], ["wall"]],
        [["wall"], [], [], ["computer"], [], ["wall"]],
        [["wall"], [], [], [], ["wall"], ["wall"]],
        [["wall"], ["wall"], ["wall"], ["wall"], ["wall"], ["wall"]],
    ]
    test = make_new_game(test_game)
    print(solve_puzzle(test))
