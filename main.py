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


def apply_ca(field):
    rfile = "rules.json"
    rules = load_rules(rfile)

    gens = 100
    ca = CellularAutomata(field, rules)

    sys.stdout.write("[%s]" % (" " * gens))
    sys.stdout.flush()
    sys.stdout.write("\b" * (gens+1))  # return to start of line, after '['

    for i in range(gens):

        ca.run()
        sys.stdout.write("-")
        sys.stdout.flush()

    sys.stdout.write("]\n")  # this ends the progress bar
    improc.save_img(RESULTS_PATH, "result.jpg", ca.field)


if __name__ == "__main__":
    imfile = "sat.jpg"
    img = load_compare(SAMPLES_PATH + imfile)
    noisy = load_noise(SAMPLES_PATH + imfile)

    improc.save_img(SAMPLES_PATH, "input.jpg", img)
    improc.save_img(SAMPLES_PATH, "input-noisy.jpg", noisy)

    apply_ca(noisy)

    err = compare_rmse(SAMPLES_PATH + "input.jpg", RESULTS_PATH + "result.jpg")
    print(err)
