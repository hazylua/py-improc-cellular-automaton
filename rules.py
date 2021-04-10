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


if __name__ == '__main__':
    patterns = generate([0, 1, 2], 8)
    group = [('identity', identity),
             ('90 degrees rotation', rotate_90_degree_clckwise),
             ('180 degrees rotation', rotate_180_degree_clckwise),
             ('270 degrees rotation', rotate_270_degree_clckwise),
             ('X axis symmetry', x_symmetry),
             ('Y axis symmetry', y_symmetry),
             ('Right diagonal symmetry', diagr_symmetry),
             ('Left diagonal symmetry', diagl_symmetry)]
    symmetries = {}

    while len(patterns) > 0:
        pattern = patterns[0]
        delete_in_place(patterns, 0)
        matrix = get_matrix(list(pattern))

        symmetries[pattern] = {}

        for action in group:
            act = action[0]
            result = get_pattern(action[1](matrix))
            symmetries[pattern][act] = result

        for key, result in symmetries[pattern].items():
            j = 0
            while j < len(patterns):
                if result == patterns[j]:
                    delete_in_place(patterns, j)
                    break
                j += 1
    
    for key in symmetries:
        print(key)
    print(len(symmetries))
