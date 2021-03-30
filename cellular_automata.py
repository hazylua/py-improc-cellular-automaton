# Set up paths for notebook
import sys
import os
    
import image_processing as improc
    
results_path = '../results/'
samples_path = '../samples/'

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
        for y in range(0, self.maxY):
            for x in range(0, self.maxX):
                neighbours = tuple(self.neighbours(x, y))
                if neighbours == self.rule:
                    field2[x][y] = 255
                    continue
                else:
                    #field2[x][y] 
                    continue
        return field2

    # ?
    def neighbours(self, x, y):
        rows = self.maxX
        cols = self.maxY if rows else 0
        for i in range(max(0, x - 1), min(rows, x + 2)):
            for j in range(max(0, y - 1), min(cols, y + 2)):
                if i == x and j == y:
                    pass
                else:
                    if self.field[x][y] == self.field[i][j]:
                        yield 'c'
                    elif self.field[x][y] > self.field[i][j]:
                        yield 0
                    else:
                        yield 255

    def run(self):
        self.tick()

# img = improc.read_preprocess(samples_path + 'white_cat.jpg')
# ca = CellularAutomata(img, (0, 0, 0, 0, 0, 0, 0 ,0))

# ca.run()
# improc.save_img(results_path, 'test.jpg', ca.field)