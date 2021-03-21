import copy
import random
import itertools as it
import image_operations

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
                neighbours = list(self.neighbours(x, y))
                
                if neighbours == self.rule:
                    field2[x][y]
                    continue

                else:
                    field2[x][y]

        return field2

    def neighbours(self, x, y):
        rows = self.maxX
        cols = self.maxY if rows else 0
        for i in range(max(0, x - 1), min(rows, x + 2)):
            for j in range(max(0, y - 1), min(cols, y + 2)):
                yield self.field[i][j]

    def run(self):
        self.tick()

def compare_all_rules():
    ret, field = image_operations.read()
    
    ruleset = list(it.product([0, 1], repeat=9))
    for i, rule in enumerate(ruleset):
        ca = CellularAutomata(field, rule)
        ca.run()
        image_operations.save(ca.field, f'./results/result_{i}.jpg')

compare_all_rules()