"""
Execute.
"""

import csv
from cellular_automata import CellularAutomata
import image_processing as improc

RESULTS_PATH = "./results/"
SAMPLES_PATH = "./samples/"


def load_image(fpath):
    """ Load image and preprocess. """
    im = improc.read_preprocess(SAMPLES_PATH + fpath)
    return im


def load_rules(rpath):
    """ Load rules from file. """
    arr = []
    with open(rpath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            arr.append(row)
    return arr


if __name__ == "__main__":
    imfile = "white_cat-noise.jpg"
    img = load_image(imfile)
    rfile = "rules.csv"
    rules = load_rules(rfile)

    pattern = list(rules[124])
    ca = CellularAutomata(img, pattern)
    ca.run()
    imresult = f"test.jpg"
    improc.save_img(RESULTS_PATH, imresult, ca.field)

    # for i, rule in enumerate(rules):
    #     pattern = list(rule[0])
    #     ca = CellularAutomata(img, pattern)
    #     for _ in range(0, 10):
    #         ca.run()
    #     imresult = f"test_{i}.jpg"
    #     improc.save_img(RESULTS_PATH, imresult, ca.field)