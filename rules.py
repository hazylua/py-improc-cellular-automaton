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
from matrix import rotate_90_degree_clckwise, x_symmetry, y_symmetry, diagl_symmetry, diagr_symmetry, show_rows, get_matrix


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
    
    patterns = generate([0, 1], 8)
    group = [rotate_90_degree_clckwise, x_symmetry, y_symmetry, diagr_symmetry, diagl_symmetry]
    
    orbits = []

    for f in group:
        flag = 0
        for pattern in patterns:
            matrix = get_matrix(list(pattern))
            action = f(matrix)
            if flag > 0 and flag < 10:
                matrix = get_matrix(list(pattern))
                action = f(matrix)
                show_rows(matrix)
                print("\nTo:\n")
                show_rows(action)
                print("\nAnd:\n")
                show_rows(matrix)
                print("#" * 50)
                flag += 1
            flag += 1
        print("*" * 100)
