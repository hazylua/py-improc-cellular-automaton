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
from skimage.metrics import structural_similarity as ssim
from sklearn.feature_selection import SequentialFeatureSelector as sfs

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


def configure_and_save(grid, rs, filename):
    h = grid.shape[0]
    w = grid.shape[1]

    ca_dimension = [h, w]
    ca_save = ImageCA(ca_dimension, grid, rs)
    ca_save.evolve(times=1)

    keys = list(ca_save.cells.keys())
    cells = list(ca_save.cells.values())

    image = []
    for row in range(0, len(keys), w):
        image_row = np.array( [ cell.state[0] for cell in cells[row:row + w] ] )
        image.append(image_row)

    save_img("./results/", filename, np.asarray(image))


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

    #im_ssim = ssim(im_compare, im_compare, data_range=im_compare.max() - im_compare.min())
    im_ssim = ssim(im_compare, im_predict, data_range=im_predict.max() - im_predict.min(), multichannel=True)
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
    _ca.evolve(times=100)

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

    # Get best values in a list.
    # First value indicates the score.
    # Second value indicates the ruleset that gave the best score.
    # Third value indicates added key to ruleset that gave the best score.
    result = [ruleset_err, ruleset, added]

    print(f'Got: {ruleset_err} from {added}')
    _ca = None
    return result

def check(img, noisy, rules, file):
    mapper = find_best
    reducer = get_best
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

    while len(rules) != 0:
        msg = f"Finding best ruleset. Number of rules: {len(rules)}"
        logger.write_to_file(msg, log)

        previous_score = best_score
        previous_ruleset = best_ruleset

        if no_change != 1:
            map_args = []
            for key in rules.keys():
                map_args.append(
                    ({**{key: rules[key]}, **best_ruleset},
                    key, img.copy(), noisy.copy())
                )

            with Pool(thread_num) as pool:
                mapped = pool.starmap(mapper, map_args)

        best_score, best_ruleset, added_key = reduce(reducer, mapped)

        if previous_score != None and previous_score > best_score:
            msg = (
                f"Previous score is better. Checking set.\n"
                f"Previous score: {previous_score}; current \"best\" score: {best_score}\n"
            )
            logger.write_to_file(msg, log)

            # Remove rule one by one to check if removing one gives a better score.
            best_ruleset_copies = [best_ruleset.copy()
                                    for _ in range(len(best_ruleset))]
            best_ruleset_keys = best_ruleset.keys()
            pairs = list(zip(best_ruleset_copies, best_ruleset_keys))
            for pair in pairs:
                    del pair[0][pair[1]]

            check_map_args = []
            for pair in pairs:
                check_map_args.append((pair[0], pair[1], img, noisy))

            with Pool(thread_num) as pool:
                    mapped = pool.starmap(mapper, check_map_args)
            
            removed_score, best_ruleset_removed, removed_key = reduce(reducer, mapped)

            msg = f"Best key removed: {removed_key}. Removed score: {removed_score}."
            logger.write_to_file(msg, log)

            # If best "removed" score is better than previous score, continue with removed score.
            if removed_score > previous_score:
                best_ruleset = best_ruleset_removed
                no_change = 0

                msg = f"Removed score without {removed_key} is better than previous score.\nReplacing and moving on."
                logger.write_to_file(msg, log)
            
            # Else continue with previous score and remove key from mapped to make it easier to check again (no need to run CA all over again), since it's the same ruleset as before and we already have the results.
            else:
                msg = f"Removed score without {removed_key} is NOT better than previous score.\nRemoving key from list of values gotten from running CA: {added_key}."
                logger.write_to_file(msg, log)

                to_remove = [best_score, best_ruleset, added_key]

                best_ruleset = previous_ruleset
                best_score = previous_score
                mapped.remove(to_remove)

                # Remove best rule from possible rules to pick.
                rules.pop(added_key, None)
                no_change = 1

        else:
            msg = (
                f"Current \"best\" ruleset is better.\n" 
                f"Best rule: {added_key}. Score: {best_score};"
            )
            logger.write_to_file(msg, log)

            msg = f"Checking set."
            logger.write_to_file(msg, log)

            best_ruleset_copies = [best_ruleset.copy()
                                    for _ in range(len(best_ruleset))]
            best_ruleset_keys = best_ruleset.keys()
            pairs = list(zip(best_ruleset_copies, best_ruleset_keys))
            for pair in pairs:
                    del pair[0][pair[1]]

            check_map_args = []
            for pair in pairs:
                check_map_args.append((pair[0], pair[1], img, noisy))

            with Pool(thread_num) as pool:
                    mapped = pool.starmap(mapper, check_map_args)
            
            removed_score, best_ruleset_removed, removed_key = reduce(reducer, mapped)

            msg = f"Best key removed: {removed_key}. Removed score: {removed_score}."
            logger.write_to_file(msg, log)
            if removed_score > best_score:
                best_ruleset = best_ruleset_removed

                msg = f"Removed score without {removed_key} is better than old score.\nReplacing and moving on.\n"
                logger.write_to_file(msg, log)
            else:
                msg = f"Removed score without {removed_key} is NOT better than previous score.\n"
                logger.write_to_file(msg, log)

            formatted_dict = ""
            for key, value in best_ruleset.items():
                formatted_dict += f"{key}: {value}\n"

            msg = f"Current best ruleset:\n{best_ruleset}\n" + ("+" * 20)
            logger.write_to_file(msg, log)

            rules.pop(added_key, None)
            no_change = 0

        configure_and_save(noisy.copy(), best_ruleset, file)

    msg = f"Finished.\nBest ruleset: {best_ruleset}."
    logger.write_to_file(msg, log)

    print("Final best ruleset:")
    for rule in best_ruleset.keys():
        print(f"{rule}")

def run():

    msg = "\n\n\n" + ("#" * 10) + "STARTING PROGRAM" + ("#" * 10) + "\n\n\n"
    logger.write_to_file(msg, log)

    # Go through all images
    img_files = load_images(compare_path, noisy_path)
    for img_file in img_files:
        rules = load_rules("rules.json")
        # rules = dict(list(rules.items())[:2])

        msg = f"Loaded {img_file}."
        logger.write_to_file(msg, log)

        img = img_files[img_file]['compare_img']
        noisy = img_files[img_file]['noisy_img']

        check(img.copy(), noisy.copy(), rules, img_file)

if __name__ == "__main__":
    thread_num = 4

    settings = load_config("settings.json")

    compare_path = settings["paths"]["samples"]["processed"]
    noisy_path = settings["paths"]["samples"]["noisy"]

    log = settings["paths"]["results"] + settings["paths"]["logs"]

    run()