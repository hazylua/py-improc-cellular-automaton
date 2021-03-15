import copy
import random

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
                neighbors = list(self.neighbours(x, y))
                
                if neighbors == rule:
                    field2[x][y] = 'o'
                    continue
                
                else:
                    field2[x][y] = ' ';
                    continue

        return field2

    def neighbours(self, x, y):
        rows = len(self.field)
        cols = len(self.field[0]) if rows else 0
        for i in range(max(0, x - 1), min(rows, x + 2)):
            for j in range(max(0, y - 1), min(cols, y + 2)):
                if (i, j) != (x, y):
                    yield self.field[i][j]

    def run(self):
        self.tick()
        
rows = 50
cols = 50
generations = 3
matrix = [ [1] * cols ] * rows

for i, row in enumerate(matrix):
    matrix[i] = 45 * ['o'] + 5 * [' ']
    random.shuffle(matrix[i])

rule = ['o'] * 8

ca = CellularAutomata(matrix, rule)

for i in range(generations):
    for i, row in enumerate(ca.field):
        for j, value in enumerate(row):
            print(ca.field[i][j], end='  ')
        print()
    ca.run()
    print('###' * cols)