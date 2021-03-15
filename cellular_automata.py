import copy
import random
import itertools as it

class CellularAutomata:
    def __init__(self, field, rule):
        self.maxX = len(field)
        self.maxY = len(field[0])
        self.field = field
        self.rule = rule

    def tick(self):
        new_field = self.tickAlgorithm()
        self.field = copy.deepcopy(new_field)
    
    def tickAlgorithm(self):
        field2 = copy.deepcopy(self.field)

        for y in range(self.maxY):
            for x in range(self.maxX):
                neighbours = tuple(self.neighbours(x, y))
                
                if neighbours == self.rule:
                    field2[x][y] = 'o'
                    continue

                else:
                    field2[x][y] = ' ';

        return field2

    def neighbours(self, x, y):
        rows = len(self.field)
        cols = len(self.field[0]) if rows else 0
        for i in range(max(0, x - 1), min(rows, x + 2)):
            for j in range(max(0, y - 1), min(cols, y + 2)):
                # if (i, j) != (x, y):
                #     yield self.field[i][j]
                yield self.field[i][j]

    def run(self):
        self.tick()

def write_field(name, matrix):    
    with open(f'{name}.txt', 'w+') as f:
        for row in matrix:
            f.write(' '.join([str(cell) for cell in row]) + '\n')
        f.close()
    
def generate_field(rows, cols, ratio):
    matrix = [ [1] * cols ] * rows
    
    dead_cells = int(cols * ratio)
    alive_cells = cols - dead_cells
    # print(dead_cells, alive_cells)
    
    for i, row in enumerate(matrix):
        matrix[i] = alive_cells * ['o'] + dead_cells * [' ']
        random.shuffle(matrix[i])
    
    return matrix

def compare_all_rules(deadc, alivec):
    field = generate_field(100, 100, 1/100)
    write_field("field", field)
    
    ruleset = list(it.product([deadc, alivec], repeat=9))
    for i, rule in enumerate(ruleset):
        ca = CellularAutomata(field, rule)
        ca.run()
        write_field(f'./results/result_{i}', ca.field)

compare_all_rules('o', ' ')