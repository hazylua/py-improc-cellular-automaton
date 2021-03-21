import copy
import random
import itertools as it

import cv2 as cv
import numpy as np

img_path = '../samples/white_cat.jpg'

cdef read():
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    binary = cv.threshold(img, 120, 255, cv.THRESH_BINARY)
    return binary

cdef save(img, path):
    cv.imwrite(path, np.float32(img))

cdef class CellularAutomata:
    cdef int maxX, maxY
    cdef unsigned char [:, :] field
    cdef rule

    def __init__(self, field, rule):
        self.maxX = len(field)
        self.maxY = len(field[0])
        self.field = field
        self.rule = rule

    def tick(self):
        new_field = self.tickAlgorithm()
        self.field = new_field
    
    cpdef unsigned char [:, :] tickAlgorithm(self):
        cdef unsigned char [:, :] field2 = self.field
        cdef int y, x
        for y in range(0, self.maxY):
            for x in range(0, self.maxX):
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
    ret, field = read()
    
    ruleset = list(it.product([0, 1], repeat=9))
    for i, rule in enumerate(ruleset):
        ca = CellularAutomata(field, rule)
        ca.run()
        save(ca.field, f'./results/result_{i}.jpg')

compare_all_rules()