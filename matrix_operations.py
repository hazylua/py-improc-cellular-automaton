def write_field(name, matrix):    
    with open(f'{name}.txt', 'w+') as f:
        for row in matrix:
            f.write(' '.join([str(cell) for cell in row]) + '\n')
        f.close()
    
def generate_field(rows, cols, ratio):
    matrix = [ [1] * cols ] * rows
    
    dead_cells = int(cols * ratio)
    alive_cells = cols - dead_cells
    
    for i, row in enumerate(matrix):
        matrix[i] = alive_cells * ['o'] + dead_cells * [' ']
        random.shuffle(matrix[i])
    
    return matrix