"""
Eliminate all symmetries and reflection.
Burnside's lemma = ( 3^8 + 4*3^5 + 2*3^2 + 3^4 ) / 8 = 954 possible patterns.
3^8 = 6561 possible patterns, in total.
"""

import itertools as it
from math import ceil


def rotate_90_degree_clckwise(matrix):
    """ Get matrix rotated 90 degrees clockwise. """
    rotated = []
    for i in range(len(matrix[0])):
        li = list(map(lambda x: x[i], matrix))
        li.reverse()
        rotated.append(li)

    return rotated


def x_simmetry(matrix):
    """ Get reflection of matrix in the x-axis. """
    sim = matrix
    start = ceil(len(matrix)/2)
    stop = len(matrix)
    reflect = start - 1 - len(matrix) % 2
    for i in range(start, stop):
        sim[i] = matrix[reflect]
        reflect -= 1
    return sim


def delete_in_place(matrix, index):
    """ Helper for 'del'. """
    #print(f'{index}: {matrix[index]}')
    del matrix[index]
    return matrix


def show_rows(matrix):
    """ Print matrix. """
    for row in matrix:
        print(row)


def generate():
    """ Generate all possible neighborhood configurations. """
    results = []
    patterns = list(it.product([0, 1, 2], repeat=8))

    # Check each pattern in the list of patterns generated to see if another pattern is a simmetry or a reflection of itself.
    # If it is, remove said simmetric/reflection matrix from the list and continue checking, else continue checking.
    i = 0
    while i < len(patterns):
        if i == 0 or i == (len(patterns) - 1):
            results.append(patterns[i])
            i += 1

        else:
            results.append(patterns[i])

            pattern = list(patterns[i])
            matrix = [pattern[0:3], [pattern[3],
                                     ' ', pattern[4]], pattern[5:8]]

            # checks
            rot90 = [[matrix[x][y] for x in range(len(matrix))] for y in range(
                len(matrix[0])-1, -1, -1)]
            rot180 = [[rot90[x][y] for x in range(len(rot90))] for y in range(
                len(rot90[0])-1, -1, -1)]
            rot270 = [[rot180[x][y] for x in range(len(rot180))] for y in range(
                len(rot180[0])-1, -1, -1)]
            
            j = i + 1
            while j < (len(patterns) - 2):
                cpattern = list(patterns[j])
                cmatrix = [cpattern[0:3], [
                    cpattern[3], ' ', pattern[4]], cpattern[5:8]]
                
                if cmatrix == rot90:
                    del patterns[j]
                elif cmatrix == rot180:
                    del patterns[j]
                elif cmatrix == rot270:
                    del patterns[j]
                    
                j += 1

            i += 1
            print(len(patterns))

            # for j in range( ( i + 1 ), ( len(patterns) - 2 ) ):
            #     cpattern = list(patterns[j])
            #     cmatrix = [cpattern[0:3], [cpattern[3], ' ', pattern[4]], cpattern[5:8]]

            #     if cmatrix == rot90:
            #         # print(rot90, '\t', cmatrix)
            #         # print('*'*50, '\n')
            #         # input('press to continue')
            #         del patterns[j]

     #print(i)
    # print(len(results))


if __name__ == '__main__':
    generate()
