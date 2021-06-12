from os import path, walk

import cv2 as cv
import numpy as np
import pandas as pd

from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

import image_processing as improc
from cellular_automaton import CAImageFilter
from setup_samples import load_config
from file_processing import clear_dir, get_list_of_files

from matplotlib import pyplot as plt


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

            # Gaussian filter
            sigma_filter = 1
            ca_result_path = path.join(
                results_path, 'gaussian_filter', type, rate, str(sigma_filter))
            gaussian_filtered = ndimage.gaussian_filter(img, sigma_filter)

            # Median filter
            median_size = 5
            ca_result_path = path.join(
                results_path, 'median_fitler', type, rate, str(sigma_filter))
            median_filtered = ndimage.median_filter(img, size=median_size)


def run_filters():

    for root, dirs, files in walk(noisy_path, topdown=False):
        for img_file in files:
            img_file_root = path.join(root, img_file)
            img_file_root = path.normpath(img_file_root)

            type = path.basename(path.abspath(path.join(root, '..')))
            rate = path.basename(root)

            img = improc.read_preprocess(img_file_root)

            # Gaussian filter
            sigma_filter = 1
            gaussian_result_path = path.join(
                results_path, 'gaussian_filter', type, rate, str(sigma_filter))
            gaussian_filtered = ndimage.gaussian_filter(
                img, sigma=sigma_filter)
            improc.save_img(gaussian_result_path, img_file, gaussian_filtered)

            # Median filter
            median_size = 5
            median_result_path = path.join(
                results_path, 'median_filter', type, rate, str(median_size))
            median_filtered = ndimage.median_filter(img, size=median_size)
            improc.save_img(median_result_path, img_file, median_filtered)


def run_comparisons():
    index = get_list_of_files(processed_path)

    values = {}
    for root, dirs, files in walk(results_path, topdown=False):
        for img_file in files:
            img_file_root = path.join(root, img_file)
            img_file_root = path.normpath(img_file_root)

            filter_type = path.basename(path.abspath(
                path.join(root, '..', '..', '..')))
            noise_type = path.basename(
                path.abspath(path.join(root, '..', '..')))
            noise_rate = path.basename(path.abspath(path.join(root, '..')))
            variable = path.basename(root)

            result = improc.read_preprocess(img_file_root)

            noisy_compare_path = path.join(
                noisy_path, noise_type, noise_rate, img_file)
            noisy = improc.read_preprocess(noisy_compare_path)

            # print(filter_type, noise_type, noise_rate, variable)
            result_mse, result_ssim = get_comparisons(noisy, result)
            result_mse = round(result_mse, 4)
            result_ssim = round(result_ssim, 4)
            print(result_mse, result_ssim)

            key_mse = (filter_type, noise_type,
                       noise_rate, variable, 'mse')
            key_ssim = (filter_type, noise_type,
                        noise_rate, variable, 'ssmi')
            if key_mse not in values:
                values[key_mse] = []
            if key_ssim not in values:
                values[key_ssim] = []
            values[key_mse].append(result_mse)
            values[key_ssim].append(result_ssim)

    df = pd.DataFrame(values, index=index)
    table_path = './table.xlsx'
    df.to_excel(table_path, encoding='utf-8')
    print(df)


def run_test():
    sample = cv.imread(
        "./test2.png", cv.IMREAD_GRAYSCALE)
    result = sample.copy()
    timelapse = sample.copy()

    for _ in range(10):
        result = ca_filter(result, 1)
        timelapse = np.concatenate((timelapse, result), axis=1)
        print(get_comparisons(sample, result))
        # result = ndimage.gaussian_filter(sample, 1)
        # result = ndimage.median_filter(sample, size=5)
    plt.imshow(timelapse, cmap="gray")
    plt.show()


if __name__ == '__main__':

    config = load_config("./settings.json")

    noisy_path = path.normpath(path.abspath(
        "./") + config["paths"]["samples"]["noisy"])
    processed_path = path.normpath(path.abspath(
        "./") + config["paths"]["samples"]["processed"])
    results_path = path.normpath(path.abspath("./results/"))

    ichoice = int(input(
        "(1) Run CA filter\n"
        "(2) Run classic filters (gaussian and median)\n"
        "(3) Clear results\n"
        "(4) Get comparisons\n"
        "(5) Run test\n"
        "Choice: "))
    print(ichoice)
    if(ichoice == 1):
        run()
    elif(ichoice == 2):
        run_filters()
    elif(ichoice == 3):
        clear_dir(results_path)
    elif(ichoice == 4):
        run_comparisons()
    elif(ichoice == 5):
        run_test()
    else:
        exit(0)

    # run_test()
