"""
Execute.
"""

import sys
import json
from sklearn.metrics import mean_squared_error
from cellular_automata import CellularAutomata
import image_processing as improc

RESULTS_PATH = "./results/"
SAMPLES_PATH = "./samples/"
STATES_PATH = "./states/"
resize = (True, 0.3)


def load_compare(fpath):
    """ Load image without noise. """

    im = improc.read_preprocess(
        fpath, resize=resize[0], height_resize=resize[1], noise=False)
    return im


def load_noise(fpath):
    """ Load image with noise. """

    im = improc.read_preprocess(
        fpath, resize=resize[0], height_resize=resize[1], noise="salt_pepper")
    return im


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

    im = load_compare(im_compare)
    im_predic = load_compare(im_predict)
    rmse = mean_squared_error(im, im_predic)
    return rmse


def get_best(val1, val2):
    """ Reducer. """

    if val1[0] > val2[0]:
        return val2
    else:
        return val1


def find_best(chunk):
    """ Mapper. """

    gens = 10

    best_err = [None, None]
    for r in chunk.keys():
        ca = CellularAutomata(noisy, {r: chunk[r][0]})
        for _ in range(gens):
            ca.run()

        rule_err = compare_rmse(img, ca.field)

        if best_err[0] is None:
            best_err = [rule_err, r]
        elif best_err[0] > rule_err:
            best_err = [rule_err, r]

    return best_err


if __name__ == "__main__":
    rfile = "rules.json"
    rules = load_rules(rfile)
    num_splits = 4

    best_rules = {}        
    while len(best_rules) < 100:
        chunks = list(split_rules(rules))
        mapper = find_best
        reducer = get_best

        with Pool(4) as pool:
            mapped = pool.map(mapper, chunks)

        best_rule = reduce(reducer, mapped)
        key = best_rule[1]

        best_rules[key] = rules[key]
        rules.pop(key, None)
