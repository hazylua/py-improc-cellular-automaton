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
from math import ceil


def rotate_90_degree_clckwise(m):
    """ Get matrix rotated 90 degrees clockwise. """
    rotated = []
    for i in range(len(m[0])):
        li = list(map(lambda x: x[i], m))
        li.reverse()
        rotated.append(li)

    return rotated


def x_simmetry(m):
    """ Get reflection of matrix in the x-axis. """
    sim = list(m)
    start = ceil(len(m)/2)
    stop = len(m)
    reflect = start - 1 - len(m) % 2
    for i in range(start, stop):
        sim[i] = m[reflect]
        reflect -= 1
    return sim


def delete_in_place(m, index):
    """ Helper for 'del'. """
    #print(f'{index}: {matrix[index]}')
    del m[index]
    return m


def show_rows(m):
    """ Print matrix. """
    for row in m:
        print(row)


def generate(values, size):
    """ Generate all possible neighborhood configurations. """
    generated = list(it.product(values, repeat=size))
    return generated


def get_matrix(p):
    m = [p[0:3],
            [p[3], ' ', p[4]],
            p[5:8]]
    return m

if __name__ == '__main__':
    values = [0, 1]
    size = 8
    patterns = generate(values, size)
    group = [rotate_90_degree_clckwise, x_simmetry]
    piss = get_matrix(list(patterns[1]))

    for f in group:
        for pattern in patterns:
            matrix = get_matrix(list(pattern))
            
        print('-'*50)
        
    
