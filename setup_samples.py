""" Generate images to process and compare. """

import json
from os import path, walk
import image_processing as improc


def save_configured():
    for root, dirs, files in walk(clean_path, topdown=False):
        for img_file in files:
            proc_path = path.join(root, img_file)
            proc_path = path.normpath(proc_path)
            print(proc_path)
            img_configured = improc.read_preprocess(
                proc_path, resize=resize, height_resize=height_max, width_resize=width_max)

            img_file_no_suffix = path.splitext(img_file)[0]
            improc.save_img(
                processed_path, f"{img_file_no_suffix}.jpg", img_configured)


def save_noisy():
    salt_pepper_rates = [0.05, 0.25, 0.50]
    gaussian_sigmas = [0.5, 0.9]

    for root, dirs, files in walk(processed_path, topdown=False):
        for img_file in files:
            proc_path = path.join(root, img_file)
            proc_path = path.normpath(proc_path)

            for st_rate in salt_pepper_rates:
                img_salt_pepper = improc.read_preprocess(
                    proc_path, noise="salt_pepper", rate=st_rate)

                st_rate_path = str(st_rate).replace('.', '_')
                save_path = path.join(noisy_path, 'salt_pepper', st_rate_path)
                improc.save_img(save_path, img_file, img_salt_pepper)

            for sigma in gaussian_sigmas:
                img_gaussian = improc.read_preprocess(
                    proc_path, noise="gaussian", rate=sigma)

                sigma_path = str(sigma).replace('.', '_')
                save_path = path.join(noisy_path, 'gaussian', sigma_path)
                improc.save_img(save_path, img_file, img_gaussian)


def load_config(path):
    config = dict()
    with open(path, "r") as f:
        config = json.load(f)
    return config


if __name__ == '__main__':
    config = load_config("./settings.json")

    resize = config["process"]["resize"]
    height_max = config["process"]["height_max"]
    width_max = config["process"]["width_max"]

    clean_path = path.normpath(path.abspath(
        "./") + config["paths"]["samples"]["clean"])
    noisy_path = path.normpath(path.abspath(
        "./") + config["paths"]["samples"]["noisy"])
    processed_path = path.normpath(path.abspath(
        "./") + config["paths"]["samples"]["processed"])

    # save_configured()
    save_noisy()
