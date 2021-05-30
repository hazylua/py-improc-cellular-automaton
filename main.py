""" Try to find best ruleset by running CA and using SFFS. """

import os
from time import time

import itertools as it
import json
from functools import reduce
from multiprocessing import Pool
from typing import Sequence

import cv2 as cv
import numpy as np
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim

import logger
from cellular_automaton import CellularAutomaton, EdgeRule, MooreNeighbourhood
from image_processing import save_img
from setup_samples import load_config


class ImageCA(CellularAutomaton):
    """
    Implementation of cellular automaton using grayscale image pixel values as initial state.
    """

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
        New state will be the average of neighbour cells.
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


def configure_and_save(grid, rs):
    hh = noisy.shape[0]
    ww = noisy.shape[1]
    ca_save = ImageCA([hh, ww], grid, rs)
    keys = list(ca_save.cells.keys())
    cells = list(ca_save.cells.values())
    image = []

    start = 0
    row_size = w - 1

    for row in range(h):
        img_row = np.array([cell.state[0]
                            for cell in cells[start:start + row_size]])
        start = start + row_size + 1
        image.append(img_row)
    save_img("./results/", "result.jpg", np.asarray(image))


def load_rules(rpath):
    """ Load rules from file. """

    arr = []
    match_list = {}
    with open(rpath, "r") as f:
        arr = json.load(f)
        for ruleset in arr:
            match_list[f'{ruleset[0]}'] = ruleset
    return match_list


def compare_ssim(im_compare, im_predict):
    """ Get RMSE of two images. Calculates the difference between two images """

    im_mse = mse(im_compare, im_predict)
    im_ssim = ssim(im_compare, im_predict)
    return im_ssim


def get_best(val1, val2):
    """ Reducer. """

    if val1[0] < val2[0]:
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

    ruleset_err = compare_ssim(img_compare, _predicted)

    result = [ruleset_err, ruleset, added]

    print(f'Got: {ruleset_err} from {added}')
    _ca = None
    return result

def load_images(c_path, n_path):
    abspath_compare = os.path.abspath("./") + c_path
    abspath_noisy = os.path.abspath("./") + n_path
    compare_imgs = os.listdir(abspath_compare)
    noisy_imgs = os.listdir(abspath_noisy)
    imgs_pair = zip(compare_imgs, noisy_imgs)
    compare_noisy_pair = {}
    for noisy, compare in imgs_pair:
        compare_noisy_pair[noisy] = {}
        compare_noisy_pair[noisy]['compare_img'] = cv.imread(abspath_compare + compare, cv.IMREAD_GRAYSCALE)
        compare_noisy_pair[noisy]['noisy_img'] = cv.imread(abspath_noisy + noisy, cv.IMREAD_GRAYSCALE)
    
    return compare_noisy_pair

if __name__ == "__main__":
    thread_num = 4

    rules = load_rules("rules.json")
    rules = dict(list(rules.items())[:16])

    settings = load_config("settings.json")
    log = settings["paths"]["results"] + settings["paths"]["logs"]
    split_size = settings["split_size"]
    compare_path = settings["paths"]["samples"]["processed"]
    noisy_path = settings["paths"]["samples"]["noisy"]

    msg = ("#" * 10) + "STARTING PROGRAM" + ("#" * 10)
    logger.write_to_file(msg, log)

    img_files = load_images(compare_path, noisy_path)
    for img_file in img_files:
        msg = f"Loaded {img_file}."
        logger.write_to_file(msg, log)

        img = img_files[img_file]['compare_img']
        noisy = img_files[img_file]['noisy_img']
    
        # Number of rows.
        h = noisy.shape[0]
        # Number of columns.
        w = noisy.shape[1]

        msg = f"\nStarting search."
        logger.write_to_file(msg, log)

        removed_value = None
        best_score = None
        best_ruleset = {}

        no_change = 0
        i = 0
        while len(best_ruleset) < 10 or no_change < 100 or len(rules) > 100:
            msg = f"Finding best ruleset. Number of rules: {len(rules)}"
            logger.write_to_file(msg, log)

            # Find rule with best score.
            previous_score = best_score
            previous_ruleset = best_ruleset

            mapper = find_best
            reducer = get_best

            map_args = []
            for key in rules.keys():
                map_args.append(
                    ({**{key: rules[key]}, **best_ruleset},
                    key, img.copy(), noisy.copy())
                )

            with Pool(thread_num) as pool:
                mapped = pool.starmap(mapper, map_args)

            # Get best values in a list.
            # First value indicates the score.
            # Second value indicates the ruleset that gave the best score.
            # Third value indicates added key to ruleset that gave the best score.
            best_score, best_ruleset, added_key = reduce(reducer, mapped)

            # Remove best rule from possible rules to pick.
            rules.pop(added_key, None)

            msg = f"Got best rule: {added_key}. Score: {best_score}; Best ruleset: {list(best_ruleset.keys())}"
            logger.write_to_file(msg, log)

            # Check if it's the first iteration.
            if len(best_ruleset) > 1:
                msg = "Checking set."
                logger.write_to_file(msg, log)

                # Get copies of best_ruleset, attach a key to remove with each and remove the key paired with it.
                best_ruleset_copies = [best_ruleset.copy()
                                    for _ in range(len(best_ruleset))]
                best_ruleset_keys = best_ruleset.keys()
                pairs = list(zip(best_ruleset_copies, best_ruleset_keys))
                for pair in pairs:
                    del pair[0][pair[1]]

                # Map of arguments for starmap.
                check_map_args = []
                for pair in pairs:
                    check_map_args.append((pair[0], pair[1], img, noisy))

                with Pool(thread_num) as pool:
                    mapped = pool.starmap(mapper, check_map_args)

                removed_score, best_ruleset_removed, removed_key = reduce(
                    reducer, mapped)

                msg = f"Best key removed: {removed_key}. Removed score: {removed_score}."
                logger.write_to_file(msg, log)

                if removed_score > best_score:
                    best_ruleset = best_ruleset_removed
                    no_change = 0

                    msg = f"Removed score without {removed_key} is better.\nMoving on and replacing..."
                    logger.write_to_file(msg, log)
                else:
                    no_change += 1
                    msg = f"No change; no_change = {no_change}. Moving on..."
                    logger.write_to_file(msg, log)

            msg = f"best_ruleset: {best_ruleset.keys()}\nFinal score: {best_score}"
            logger.write_to_file(msg, log)

            if(len(best_ruleset) > 2):
                configure_and_save(noisy.copy(), best_ruleset)

        print("Final best ruleset:")
        for rule in best_ruleset.keys():
            print(f"{rule}")

