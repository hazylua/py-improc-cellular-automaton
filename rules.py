"""
Eliminate all symmetries and reflection.
Burnside's lemma = ( 3^8 + 4*3^5 + 2*3^2 + 3^4 ) / 8 = 954 possible patterns.
3^8 = 6561 possible patterns, in total.
Check each pattern in the list of patterns generated
    to see if another pattern is a simmetry or a reflection of itself.
If it is, remove said simmetric/reflection matrix
    from the list and continue checking, else continue checking.
"""

import itertools as it
import json
from matrix import rotate_90_degree_clckwise, x_symmetry, y_symmetry, diagl_symmetry, diagr_symmetry, show_rows, get_matrix, get_pattern, rotate_180_degree_clckwise, rotate_270_degree_clckwise, identity


def delete_in_place(values, pos):
    """ Helper for 'del'. """
    #print(f'{pos}: {values[pos]}')
    del values[pos]
    return values


def generate(array, size):
    """ Generate all possible neighborhood configurations. """
    generated = list(it.product(array, repeat=size))
    return generated


def get_rules(arr, s):
    patterns = generate(arr, s)
    group = [identity, rotate_90_degree_clckwise, rotate_180_degree_clckwise,
             rotate_270_degree_clckwise, x_symmetry, y_symmetry, diagr_symmetry, diagl_symmetry]

    symmetries = []

    i = 0
    while len(patterns) > 0:
        pattern = patterns[0]
        delete_in_place(patterns, 0)

        matrix = get_matrix(list(pattern))

        symmetries.append([])

        for action in group:
            act = action(matrix)
            act_pattern = get_pattern(act)
            symmetries[i].append(act_pattern)

        for result in symmetries[i]:
            j = 0
            while j < len(patterns):
                if result == patterns[j]:
                    delete_in_place(patterns, j)
                    break
                j += 1
        i += 1

    return symmetries


if __name__ == "__main__":
    states = []
    neighbourhood_size = 0

    with open("./settings.json") as f:
        settings = json.load(f)
        states = settings['states']
        neighbourhood_size = settings['neighbourhood_size']

    rules = get_rules(states, neighbourhood_size)
    # print(len(rules))

    with open("rules.json", "w+", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=4)

    # with open("rules.csv", "r") as f:
    #     reader = csv.reader(f)
    #     arr = []
    #     for row in reader:
    #         print(row)
    #         arr.append(row)
    #     print(len(arr))
