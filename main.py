"""
Execute.
"""

import json
from cellular_automata import CellularAutomata
import image_processing as improc

RESULTS_PATH = "./results/"
SAMPLES_PATH = "./samples/"


def load_image(fpath):
    """ Load image and preprocess. """
    im = improc.read_preprocess(
        SAMPLES_PATH + fpath, resize=True, height_resize=0.7)
    return im


def load_rules(rpath):
    """ Load rules from file. """
    arr = []
    with open(rpath, "r") as f:
        arr = json.load(f)
    return arr


if __name__ == "__main__":
    imfile = "sat.jpg"
    img = load_image(imfile)
    improc.save_img(SAMPLES_PATH, "sat-salted.jpg", img)
    rfile = "rules.json"
    rules = load_rules(rfile)

    pattern = rules[120][0]
    ca = CellularAutomata(img, pattern)
    for _ in range(0, 20):
        ca.run()
    imresult = f"test.jpg"
    improc.save_img(RESULTS_PATH, imresult, ca.field)

    # for i, rule in enumerate(rules):
    #     pattern = rule[0]
    #     ca = CellularAutomata(img, pattern)
    #     for _ in range(0, 10):
    #         ca.run()
    #     imresult = f"test_{i}.jpg"
    #     improc.save_img(RESULTS_PATH, imresult, ca.field)
