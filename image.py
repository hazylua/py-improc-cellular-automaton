from os import path, walk

from typing import Sequence
from collections import Counter

import cv2 as cv
import numpy as np

from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

import image_processing as improc
from cellular_automaton import CAImageFilter
from setup_samples import load_config

from matplotlib import pyplot as plt


def get_comparisons(im_compare, im_predict):
    im_mse = mse(im_compare, im_predict)
    im_ssim = ssim(im_compare, im_predict, data_range=im_predict.max(
    ) - im_predict.min(), multichannel=True)

    return [im_mse, im_ssim]


def ca_filter(grid, rs, filename, save_path, gens):
    h = grid.shape[0]
    w = grid.shape[1]

    ca_dimension = [h, w]
    ca_save = CAImageFilter(ca_dimension, grid, rs)

    print("Generations: ", gens)
    ca_save.evolve(gens)

    keys = list(ca_save.cells.keys())
    cells = list(ca_save.cells.values())

    image = []
    for row in range(0, len(keys), w):
        image_row = np.array([cell.state[0]
                              for cell in cells[row:row + w]])
        image.append(image_row)
    image = np.asarray(image)

    # save_gen_path = path.join(save_path, str(gens))
    # improc.save_img(save_gen_path, filename, image)
    # comparisons.append(get_comparisons(grid, image))

    ca_save = None
    return image


def run():
    gens = int(input("Generations of CA: "))

    for root, dirs, files in walk(noisy_path, topdown=False):
        for img_file in files:
            img_file_root = path.join(root, img_file)
            img_file_root = path.normpath(img_file_root)

            type = path.basename(path.abspath(path.join(root, '..')))
            rate = path.basename(root)

            img = improc.read_preprocess(img_file_root)

            ca_result_path = path.join(results_path, 'ca', type, rate)
            ca_filter(img, {}, img_file, ca_result_path, gens)


if __name__ == '__main__':
    config = load_config("./settings.json")

    noisy_path = path.normpath(path.abspath(
        "./") + config["paths"]["samples"]["noisy"])
    processed_path = path.normpath(path.abspath(
        "./") + config["paths"]["samples"]["processed"])
    results_path = path.normpath("./results/")

    # run()

    sample = cv.imread(
        "./samples/noisy/salt_pepper/0_25/med_chest_ct.jpg", cv.IMREAD_GRAYSCALE)

    result = ca_filter(sample, {}, '', '', 100)
    plt.imshow(result, cmap="gray")
    plt.show()

    # configure_and_save(noisy.copy(), {}, 'none.jpg', 100)
