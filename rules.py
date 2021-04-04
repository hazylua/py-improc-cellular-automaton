"""
Consider simmetries: Apply rule even if pattern is rotated.
Apply Burnside's lemma to find all possible rules after eliminating symmetries and reflections. for 3 states and neighbors of 8 -> ( 3^8 + 4*3^5 + 2*3^2 + 3^4 ) / 8 = 954 possible patterns.
"""

import itertools as it

def generate():
    """ Generate patterns with diagonal simmetries. """   
    p = list(it.product([0, 1, 2], repeat=3))
    m = list(it.product([0, 1, 2], repeat=5))
    results = []
    for x in p:
        for y in m:
            row1 = [y[0], y[1], y[2]]
            row2 = [x[0], 9, y[3]]
            row3 = [x[1], x[2], y[4]]
            matrix = [row1, row2, row3]
            results.append(matrix)
    print(results[-1])

if __name__ == '__main__':
    generate()