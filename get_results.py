""" Run file and get results from menu. """

from cellular_automaton.models import CAImageFilterMedian
import sys
from os import path, walk

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim

import image_processing as improc
from cellular_automaton import CAImageFilter
from file_processing import clear_dir, get_list_of_files
from setup_samples import load_config


def get_comparisons(im_compare, im_predict):
    im_mse = mse(im_compare, im_predict)
    im_ssim = ssim(im_compare, im_predict, data_range=im_predict.max(
    ) - im_predict.min())

    return [im_mse, im_ssim]


def ca_filter(grid, gens):
    h = grid.shape[0]
    w = grid.shape[1]

    ca_dimension = [h, w]

    empty_ruleset = {}
    ca_save = CAImageFilter(ca_dimension, grid, empty_ruleset)

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

            # CA filter
            ca_result_path = path.join(
                results_path, 'ca', type, rate, str(gens))
            ca_filtered = ca_filter(img, gens)
            improc.save_img(ca_result_path, img_file, ca_filtered)


def ca_filter_median(grid, gens):
    h = grid.shape[0]
    w = grid.shape[1]

    ca_dimension = [h, w]

    empty_ruleset = {}
    ca_save = CAImageFilterMedian(ca_dimension, grid, empty_ruleset)

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


def run_ca_median():
    gens = int(input("Generations of CA: "))

    for root, dirs, files in walk(noisy_path, topdown=False):
        for img_file in files:
            img_file_root = path.join(root, img_file)
            img_file_root = path.normpath(img_file_root)

            filter_type = path.basename(path.abspath(path.join(root, '..')))
            rate = path.basename(root)

            img = improc.read_preprocess(img_file_root)

            # CA median filter
            ca_result_path = path.join(
                results_path, 'ca_median', filter_type, rate, str(gens))
            ca_filtered = ca_filter_median(img, gens)
            improc.save_img(ca_result_path, img_file, ca_filtered)


def run_filters():

    for root, dirs, files in walk(noisy_path, topdown=False):
        for img_file in files:
            img_file_root = path.join(root, img_file)
            img_file_root = path.normpath(img_file_root)

            noise_type = path.basename(path.abspath(path.join(root, '..')))
            rate = path.basename(root)

            img = improc.read_preprocess(img_file_root)

            # Gaussian filter
            sigmas = range(1, 11, 1)
            for sigma in sigmas:
                # sigma_filter = 1
                gaussian_result_path = path.join(
                    results_path, 'gaussian_filter', noise_type, rate, str(sigma))
                gaussian_filtered = ndimage.gaussian_filter(
                    img, sigma=sigma)
                improc.save_img(gaussian_result_path,
                                img_file, gaussian_filtered)

            # Median filter
            sizes = range(1, 11, 1)
            for size in sizes:
                # median_size = 5
                median_result_path = path.join(
                    results_path, 'median_filter', noise_type, rate, str(size))
                median_filtered = ndimage.median_filter(img, size=size)
                improc.save_img(median_result_path, img_file, median_filtered)


def run_comparisons():
    index = get_list_of_files(processed_path)
    index_check = []

    values_ssim = {}
    values_rmse = {}
    for root, dirs, files in walk(results_path, topdown=False):
        for img_file in files:
            if img_file not in index_check:
                index_check.append(img_file)
            img_file_root = path.join(root, img_file)
            img_file_root = path.normpath(img_file_root)

            filter_type = path.basename(path.abspath(
                path.join(root, '..', '..', '..')))
            noise_type = path.basename(
                path.abspath(path.join(root, '..', '..')))
            noise_rate = path.basename(path.abspath(path.join(root, '..')))
            variable = path.basename(root)

            result = improc.read_preprocess(img_file_root)

            compare_path = path.join(
                processed_path, img_file)
            compare = improc.read_preprocess(compare_path)

            result_rmse, result_ssim = get_comparisons(compare, result)
            result_rmse = round(result_rmse, 4)
            result_ssim = round(result_ssim, 4)
            print(img_file, result_rmse, result_ssim)

            key_mse = (filter_type, noise_type,
                       noise_rate, variable)
            key_ssim = (filter_type, noise_type,
                        noise_rate, variable)

            if key_mse not in values_rmse:
                values_rmse[key_mse] = []
            if key_ssim not in values_ssim:
                values_ssim[key_ssim] = []
            values_rmse[key_mse].append(result_rmse)
            values_ssim[key_ssim].append(result_ssim)

    df_rmse = pd.DataFrame(values_rmse, index=index)
    df_ssim = pd.DataFrame(values_ssim, index=index)
    table_ssim_path = './table_ssim.xlsx'
    table_rmse_path = './table_rmse.xlsx'
    df_rmse.to_excel(table_rmse_path, encoding='utf-8')
    df_ssim.to_excel(table_ssim_path, encoding='utf-8')

    print(index, '#'*50, index_check, '#'*50)
    print(df_rmse, '*'*50, df_ssim)


if __name__ == '__main__':

    config = load_config("./settings.json")

    noisy_path = path.normpath(path.abspath(
        "./") + config["paths"]["samples"]["noisy"])
    processed_path = path.normpath(path.abspath(
        "./") + config["paths"]["samples"]["processed"])
    results_path = path.normpath(path.abspath(
        "./") + config["paths"]["results"])

    ichoice = int(input(
        "(1) Run CA filter\n"
        "(2) Run classic filters (gaussian and median)\n"
        "(3) Clear results\n"
        "(4) Get comparisons\n"
        "(5) Run test\n"
        "(6) Run CA median filter\n"
        "Choice: "))
    print(ichoice)
    if ichoice == 1:
        run()
    elif ichoice == 2:
        run_filters()
    elif ichoice == 3:
        clear_dir(results_path)
    elif ichoice == 4:
        run_comparisons()
    elif ichoice == 5:
        run_test()
    elif ichoice == 6:
        run_ca_median()
    else:
        sys.exit()
