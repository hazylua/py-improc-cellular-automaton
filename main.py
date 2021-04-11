"""
Execute.
"""

from cellular_automata import CellularAutomata
import image_processing as improc
from rules import get_rules

RESULTS_PATH = "./results/"
SAMPLES_PATH = "./samples/"

if __name__ == "__main__":
    imfile = "white_cat-noise.jpg"
    
    arr = [0, 1, 2]
    neighborhood = 8
    rules = get_rules(arr, neighborhood)
    
    img = improc.read_preprocess(SAMPLES_PATH + imfile)
    for i, rule in enumerate(rules):
        pattern = list(rule[0])
        ca = CellularAutomata(img, pattern)
        for _ in range(0, 10):
            ca.run()
        imresult = f"test_{i}.jpg"
        improc.save_img(RESULTS_PATH, imresult, ca.field)
