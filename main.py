"""
Execute.
"""

import json
from cellular_automata import CellularAutomata
import image_processing as improc

RESULTS_PATH = "./results/"
SAMPLES_PATH = "./samples/"
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


    rfile = "rules.json"
    rules = load_rules(rfile)

    for i, rule in enumerate(rules):
        pattern = rule[0]
        ca = CellularAutomata(img, pattern)
        for j in range(0, 20):
            print(f"{i}:{j} - RULE: {pattern} - Running...")
            ca.run()
        imresult = f"result_{i}.jpg"
        improc.save_img(RESULTS_PATH, imresult, ca.field)
