""" Run cellular automata. """

import json
from typing import Sequence
from itertools import islice

import cv2 as cv
import numpy as np

from sklearn.metrics import mean_squared_error

from cellular_automaton import CellularAutomaton, MooreNeighbourhood, EdgeRule
from process_images import load_config


def load_rules(rpath):
    """ Load rules from file. """

    arr = []
    match_list = {}
    with open(rpath, "r") as f:
        arr = json.load(f)
        for ruleset in arr:
            match_list[f'{ruleset[0]}'] = ruleset
    return match_list


def compare_rmse(im_compare, im_predict):
    """ Get RMSE of two images. """

    rmse = mean_squared_error(im_compare, im_predict)
    return rmse


def chunks(data, size=10000):
    """ Split dict into chunks. """

    it = iter(data)
    for _ in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


def get_best(val1, val2):
    """ Reducer. """

    if val1[0] > val2[0]:
        return val2
    else:
        return val1


def find_best(chunk, img_compare, img_noisy):
    """ Mapper. """

    print("Starting.")
    compare = np.copy(img_compare)
    field = np.copy(img_noisy)

    gens = 10

    best_err = [None, None]
    for r in chunk.keys():
        ca = CellularAutomata(field, {r: chunk[r][0]})
        for _ in range(gens):
            ca.evolve()

        rule_err = compare_rmse(compare, ca.field)

        if best_err[0] is None:
            best_err = [rule_err, r]
        elif best_err[0] > rule_err:
            best_err = [rule_err, r]

    print(f'Found best: {best_err}')
    return best_err


class ImageCA(CellularAutomaton):
    """ Use Cellular Automaton with grayscale image pixel values as initial state. """

    def __init__(self, dimension, image, ruleset):
        super().__init__(dimension=dimension, image=image, ruleset=ruleset,
                         neighbourhood=MooreNeighbourhood(EdgeRule.IGNORE_EDGE_CELLS))

    def init_cell_state(self, cell_coordinate: Sequence) -> Sequence:  # pragma: no cover
        x, y = cell_coordinate
        init = self._image[x][y]
        return [init]

    def evolve_rule(self, last_cell_state, neighbours_last_states):
        """
        Change cell state if neighbours match a rule in ruleset.
        New state will the average of neighbour cells.
        """

        new_cell_state = last_cell_state
        if neighbours_last_states == []:
            return new_cell_state

        thresholded = self.__local_threshold(
            neighbours_last_states, last_cell_state)
        hashed = f'{thresholded}'

        if self.ruleset.get(hashed):
            new_cell_state = self.__local_average(neighbours_last_states)
            # print(last_cell_state, new_cell_state)
            return [new_cell_state]
        else:
            return new_cell_state

    @staticmethod
    def __local_average(neighbours):
        """ Get local average of neighbours. """

        values = list(map(sum, neighbours))
        local = int(sum(values) / len(values))
        return local

    @staticmethod
    def __local_threshold(neighbours, c):
        """ Get threhsolded neighbours based on central cell. """

        return [0 if n[0] > c[0] else 1 if n[0] == c[0] else 2 for n in neighbours]


if __name__ == "__main__":
    rules = load_rules("rules.json")
    settings = load_config("settings.json")
    split_size = settings["split_size"]
    compare_path = settings["paths"]["samples"]["processed"]
    noisy_path = settings["paths"]["samples"]["noisy"]

    # Load as grayscale image.
    fpath = "satellite_4.jpg"
    img = cv.imread(compare_path + fpath, cv.IMREAD_GRAYSCALE)
    noisy = cv.imread(noisy_path + fpath, cv.IMREAD_GRAYSCALE)

    w = noisy.shape[0]
    h = noisy.shape[1]
    ca = ImageCA(dimension=[w, h], image=noisy.tolist(), ruleset=rules)
    # CAWindow(ca, window_size=(1500, 1000)).run(evolutions_per_second=40)
    ca.evolve(times=5)
