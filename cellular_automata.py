# Set up paths for notebook
import sys
import os
import copy

class CellularAutomata:
    def __init__(self, field, rule):
        self.maxX = len(field)
        self.maxY = len(field[0])
        self.field = field
        self.rule = rule

    def tick(self):
        new_field = self.tickAlgorithm()
        self.field = new_field

    def tickAlgorithm(self):
        # deepcopy?
        field2 = self.field
        for y in range(1, self.maxY - 1):
            for x in range(1, self.maxX - 1):
                neighbours = list(self.neighbours(x, y))
                th = neighbours.pop()
                if neighbours == self.rule:
                    field2[x][y] = th
                    continue
                else:
                    continue
        return field2

    # ?
    def neighbours(self, x, y):
        rows = self.maxX
        cols = self.maxY if rows else 0
        local = 0
        for i in range(max(0, x - 1), min(rows, x + 2)):
            for j in range(max(0, y - 1), min(cols, y + 2)):
                if i == x and j == y:
                    pass
                else:
                    local += self.field[x][y]
                    if self.field[x][y] == self.field[i][j]:
                        yield 2
                    elif self.field[x][y] > self.field[i][j]:
                        yield 0
                    else:
                        yield 1
        th = int(local/8)
        yield th
                    

    def run(self):
        self.tick()