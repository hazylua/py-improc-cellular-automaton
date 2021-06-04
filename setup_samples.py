""" Generate images to process and compare. """

import json
from os import listdir, path
import image_processing as improc


def save_configured():
    clean_imgs = listdir(clean_path)

    for img_file in clean_imgs:
        print(img_file)
        img_configured = improc.read_preprocess(
            clean_path + img_file, resize=resize, height_resize=height_max, width_resize=width_max)
        improc.save_img(
            processed_path, f"{img_file}", img_configured)


def save_noisy():
    processed_imgs = listdir(processed_path)

    salt_pepper = []
    gaussian = []
    for img_file in processed_imgs:
        img_salt_pepper = improc.read_preprocess(
            processed_path + img_file, noise="salt_pepper", rate=0.09)
        salt_pepper.append(img_salt_pepper)
        img_gaussian = improc.read_preprocess(
            processed_path + img_file, noise="gaussian", rate=0.1)
        gaussian.append(img_gaussian)

    for idx, img_file in enumerate(processed_imgs):
        improc.save_img(noisy_path + 'salt_pepper/',
                        f"{img_file}", salt_pepper[idx])
        improc.save_img(noisy_path + 'gaussian/',
                        f"{img_file}", gaussian[idx])


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

    processed_path = path.abspath(
        "./") + config["paths"]["samples"]["processed"]
    clean_path = path.abspath("./") + config["paths"]["samples"]["clean"]
    noisy_path = path.abspath("./") + config["paths"]["samples"]["noisy"]
    processed_path = path.abspath(
        "./") + config["paths"]["samples"]["processed"]

    save_configured()
    save_noisy()
