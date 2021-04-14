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
        self.field = np.copy(new_field)

    def tick_algorithm(self):
        """ Define tick. """

        field2 = np.copy(self.field)
        x, y = np.meshgrid(np.arange(field2.shape[0]),
                            np.arange(field2.shape[1]), indexing='ij')
        update = np.frompyfunc(self.update_pixel, 3, 1)
        field2 = update(field2, x, y)

        return field2

    def update_pixel(self, p, x, y):
        """ Update function. """

        if x == 0 or y == 0 or x == self.maxX or y == self.maxY:
            return p
        else:
            neighbours = self.neighbours(x, y)

            if neighbours == self.rule:
                neighbours_average = self.local_average(x, y)
                return neighbours_average
            else:
                return p

    def neighbours(self, x, y):
        """ Get neighbours on postion (x, y). """

        width = self.maxX
        height = self.maxY
        neighbours = [self.local_threshold(self.field[x2][y2], self.field[x][y])
                      for x2 in range(max(0,  x-1), min(width, x+2))
                      for y2 in range(max(0, y-1), min(height, y+2)) if (x2, y2) != (x, y)]
        return neighbours

    def local_average(self, x, y):
        """ Get local average of neighbours. """

        width = self.maxX
        height = self.maxY
        neighbours = [self.field[x2][y2] for x2 in range(max(0, x-1), min(width, x+2))
                      for y2 in range(max(0, y-1), min(height, y+2)) if (x2, y2) != (x, y)]
        local = int(sum(neighbours) / len(neighbours))
        return local

    def local_threshold(self, value, compare):
        """ Threshold neighbours based on central cell. """

        if value == compare:
            return 2
        elif value < compare:
            return 0
        else:
            return 1

    def run(self):
        """ Run algorithm. """

        self.tick()
