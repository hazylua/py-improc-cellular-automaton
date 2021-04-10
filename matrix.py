""" Matrix operations for square matrices. """

from math import ceil
# import copy


def identity(m):
    """ Get the matrix itself. For labeling purposes. """
    return m


def rotate_90_degree_clckwise(matrix):
    """ Get a matrix rotated 90 degrees clockwise. """
    rot_matrix = []
    for i in range(len(matrix[0])):
        li = list(map(lambda x: x[i], matrix))
        li.reverse()
        rot_matrix.append(li)

    return rot_matrix


def rotate_180_degree_clckwise(matrix):
    """ Get a matrix rotated 180 degrees clockwise. """
    rot_90 = rotate_90_degree_clckwise(matrix)
    rot_180 = rotate_90_degree_clckwise(rot_90)
    return rot_180


def rotate_270_degree_clckwise(matrix):
    """ Get a matrix rotated 270 degrees clockwise. """
    rot_90 = rotate_90_degree_clckwise(matrix)
    rot_180 = rotate_90_degree_clckwise(rot_90)
    rot_270 = rotate_90_degree_clckwise(rot_180)
    return rot_270


def x_symmetry(matrix):
    """ Get reflection of matrix in the x-axis. """
    sym_matrix = [x[:] for x in matrix]
    height = len(matrix)
    start = ceil(height/2)
    reflect = start - 1 - height % 2
    for i in range(start, height):
        temp = matrix[i]
        sym_matrix[i] = matrix[reflect]
        sym_matrix[reflect] = temp
        reflect -= 1
    return sym_matrix


def y_symmetry(matrix):
    """ Get reflection of matrix in the y-axis. """
    # sym_matrix = copy.deepcopy(matrix)
    sym_matrix = [x[:] for x in matrix]
    width = len(sym_matrix[0])
    start = ceil(width/2)
    reflect = start - 1 - width % 2
    for i in range(start, width):
        temp = get_column(matrix, i)
        reflect_column = get_column(sym_matrix, reflect)
        sym_matrix = set_column(sym_matrix, i, reflect_column)
        sym_matrix = set_column(sym_matrix, reflect, temp)
        reflect -= 1
    return sym_matrix


def diagr_symmetry(matrix):
    """ Get reflection of matrix in line y = x. """
    sym_matrix = [x[:] for x in matrix]
    width = len(sym_matrix[0])
    height = len(sym_matrix)

    row_start = 1
    row_stop = height

    rcount = 1

    for i in range(row_start, row_stop):
        column_start = width - rcount
        column_stop = width
        ccount = 1
        for j in range(column_start, column_stop):
            temp = sym_matrix[i][j]
            sym_matrix[i][j] = matrix[i - ccount][j - ccount]
            sym_matrix[i - ccount][j - ccount] = temp
            ccount += 1
        rcount += 1
    return sym_matrix


def diagl_symmetry(matrix):
    """ Get reflection of matrix in line y = -x. """
    sym_matrix = [x[:] for x in matrix]
    height = len(sym_matrix)

    row_start = 1
    row_stop = height

    rcount = 1

    for i in range(row_start, row_stop):
        column_start = 0
        column_stop = rcount
        for j in range(column_start, column_stop):
            temp = sym_matrix[i][j]
            sym_matrix[i][j] = matrix[j][i]
            sym_matrix[j][i] = temp
        rcount += 1
    return sym_matrix


def show_rows(matrix):
    """ Print matrix. """
    string = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*string)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in string]
    print('\n'.join(table))


def get_matrix(array):
    """ Return 3x3 matrix based on pattern array. """
    return [array[0:3], [array[3], ' ', array[4]], array[5:8]]


def get_column(matrix, i):
    """ Return column of matrix. """
    return [row[i] for row in matrix]


def set_column(m, i, v):
    """ Return matrix with a replaced column. """
    for x, row in enumerate(m):
        row[i] = v[x]
    return m


def get_pattern(matrix):
    """ Get matrix in the form of a tuple. """
    pattern = []
    for row in matrix:
        for value in row:
            if value != ' ':
                pattern.append(value)
    return tuple(pattern)
