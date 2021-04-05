"""
Eliminate all symmetries and reflection.
Burnside's lemma = ( 3^8 + 4*3^5 + 2*3^2 + 3^4 ) / 8 = 954 possible patterns.
"""

import itertools as it
from numpy import tril, triu, array_equal

# needs to be implemented
def generate():
    """ Generate all possible neighborhood configurations. """
    patterns = list(it.product([0, 1, 2], repeat=8))
    temp = patterns
    i = 0
    while i < len(patterns):
        results.append(patterns[i])
        for j in range(i + 1, len(patterns) - 1):
            # check for simmetry and reflection. idea:
            # if patterns[j] is simmetry or reflection of patterns[i] then remove pattern of patterns
        # update pattern length
        # check next pattern
    

if __name__ == '__main__':
    generate()