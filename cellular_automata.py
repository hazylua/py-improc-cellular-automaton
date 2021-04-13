"""
Set up paths for notebook.
"""

import numpy as np

class CellularAutomata:
    """ Cellular automata class. """
    
    def __init__(self, field, rule):
        self.maxX = len(field)
        self.maxY = len(field[0])
        self.field = np.copy(field)
        self.rule = rule

    def tick(self):
        """ Tick. """
        new_field = self.tick_algorithm()
        self.field = new_field

    def tick_algorithm(self):
        """ Define tick. """
        field2 = np.copy(self.field)
        for y in range(1, self.maxY - 1):
            for x in range(1, self.maxX - 1):
                neighbours = list(self.neighbours(x, y))
                th = neighbours.pop()
                # print(neighbours, self.rule)
                if neighbours == self.rule:
                    # print(neighbours, self.rule)
                    # print(field2[x][y], self.field[x][y], th)
                    field2[x][y] = th
                    # print(field2[x][y], self.field[x][y], th)
                    # input()
                    continue
                else:
                    continue
        return field2

    def neighbours(self, x, y):
        """ Get neighbours on postion (x, y). """
        rows = self.maxX
        cols = self.maxY if rows else 0
        local = 0
        for i in range(max(0, x - 1), min(rows, x + 2)):
            for j in range(max(0, y - 1), min(cols, y + 2)):
                if i == x and j == y:
                    pass
                else:
                    local += self.field[i][j]
                    if self.field[x][y] == self.field[i][j]:
                        yield 2
                    elif self.field[x][y] > self.field[i][j]:
                        yield 0
                    else:
                        yield 1
        th = int(local/8)
        yield th
            

    def run(self):
        """ Run algorithm. """
        print('Running...')
        self.tick()
