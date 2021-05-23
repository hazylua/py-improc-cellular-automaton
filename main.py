""" Try to find best ruleset by running CA and using SFFS. """

import itertools as it
import json
from functools import reduce
from multiprocessing import Pool
from typing import Sequence

import cv2 as cv
import numpy as np
from sklearn.metrics import mean_squared_error

import logger
from cellular_automaton import CellularAutomaton, EdgeRule, MooreNeighbourhood
from image_processing import save_img
from setup_samples import load_config




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


def get_best(val1, val2):
    """ Reducer. """

    if val1[0] > val2[0]:
        return val2
    else:
        return val1


def find_best(ruleset, added, img_compare, img_noisy):
    """ Mapper. """
    print("Starting.")

    # Number of rows.
    height = img_noisy.shape[0]
    # Number of columns.
    width = img_noisy.shape[1]

    _ca = ImageCA(dimension=[height, width],
                  image=img_noisy.tolist(), ruleset=ruleset)
    _ca.evolve(times=10)

    _cells = [cell.state[0] for cell in _ca.cells.values()]

    _start = 0
    _row_size = width

    _img_proc = []
    for _row in range(height):
        _img_row = [cell for cell in _cells[_start:_start + _row_size]]
        _start = _start + _row_size
        _img_proc.append(_img_row)

    _predicted = np.asarray(_img_proc, dtype=np.uint8)

    ruleset_err = compare_rmse(img_compare, _predicted)

    result = [ruleset_err, ruleset, added]

    print(f'Got: {ruleset_err} from {added}')
    _ca = None
    return result


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

    def __del__(self):
        coordinates = self._current_state.keys()
        for coordinate, cell_c, cell_n in zip(coordinates, self._current_state.values(), self._next_state.values()):
            cell_c.neighbours = (None, )
            cell_n.neighbours = (None, )


if __name__ == "__main__":
    rules = load_rules("rules.json")

    log = "log.txt"
    settings = load_config("settings.json")
    split_size = settings["split_size"]
    compare_path = settings["paths"]["samples"]["processed"]
    noisy_path = settings["paths"]["samples"]["noisy"]

    # Load as grayscale image.
    fpath = "satellite_4.jpg"
    img = cv.imread(compare_path + fpath, cv.IMREAD_GRAYSCALE)
    noisy = cv.imread(noisy_path + fpath, cv.IMREAD_GRAYSCALE)

    # Number of rows.
    h = noisy.shape[0]
    # Number of columns.
    w = noisy.shape[1]

    previous_score = None
    best_ruleset = {}
    no_change = 0
    i = 0

    logger.write_to_file("*"*50, log)
    logger.write_to_file("Starting search.", log)
    while len(best_ruleset) < 100 or no_change < 10 or len(rules) > 100:
        logger.write_to_file(
            f"Finding best ruleset. Number of rules: {len(rules)}", log)
        # Find rule with best score.

        mapper = find_best
        reducer = get_best

        map_args = []
        for key in rules.keys():
            map_args.append(
                ({**{key: rules[key]}, **best_ruleset},
                 key, img.copy(), noisy.copy())
            )

        # mapped = it.starmap(mapper, map_args)

        with Pool(3) as pool:
            mapped = pool.starmap(mapper, map_args)

        # Get best values in a list.
        # First value indicates the score.
        # Second value indicates the ruleset that gave the best score.
        # Third value indicates added key to ruleset that gave the best score.
        best_score, best_ruleset, added_key = reduce(reducer, mapped)
        print(f"Got best rule: {added_key}. Score: {best_score}")
        logger.write_to_file(
            f"Got best rule: {added_key}. Score: {best_score}", log)

        # print("Ruleset:")
        # for rule in best_ruleset: print(f"{rule}")

        # Remove best rule from possible rules to pick.
        rules.pop(added_key, None)

        # Check if it's the first iteration.
        if len(best_ruleset) > 1:
            print("Running SFFS.")
            logger.write_to_file("Running SFFS.", log)
            # Remove rules to check which provides with the best result.
            # If a rule removed is better than the previous best.
            best_ruleset_copies = [best_ruleset.copy()
                                   for _ in range(len(best_ruleset))]
            best_ruleset_keys = best_ruleset.keys()
            # Copy of best_ruleset with each copy having a key removed to check which key removed gives the best value.
            pairs = list(zip(best_ruleset_copies, best_ruleset_keys))
            # Removing keys of each key.
            for pair in pairs:
                del pair[0][pair[1]]

            map_args = []
            for pair in pairs:
                map_args.append((pair[0], pair[1], img, noisy))

            with Pool(3) as pool:
                mapped = pool.starmap(mapper, map_args)

            removed_score, best_ruleset_removed, removed_key = reduce(
                reducer, mapped)

            print(
                f"Found. Best key removed: {removed_key}. Removed score: {removed_score}.")
            logger.write_to_file(
                f"Found. Best key removed: {removed_key}. Removed score: {removed_score}.", log)

            if removed_score < best_score:
                best_ruleset = best_ruleset_removed
                no_change = 0

                print("Removed score is better. Moving on and replacing...")
                logger.write_to_file(
                    "Removed score is better. Moving on and replacing...", log)
            else:
                no_change += 1
                print("No change. Moving on...")
                logger.write_to_file(
                    f"No change; no_change = {no_change}. Moving on...", log)
                continue

            print(f"Got ruleset: {best_ruleset}")

        print(f"Final score: {best_score}")
        logger.write_to_file(f"Final score: {best_score}", log)
        print("Final best ruleset:")
        for rule in best_ruleset:
            print(f"{rule}")
