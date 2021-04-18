import json
from os import listdir
import image_processing as improc


def save_configured(config):
    resize = config["process"]["resize"]
    resize_ratio = config["process"]["resize_ratio"]

    processed_path = config["paths"]["samples"]["processed"]
    clean_path = config["paths"]["samples"]["clean"]

    clean_imgs = listdir(clean_path)
    for img_file in clean_imgs:
        print(img_file)
        img_configured = improc.read_preprocess(
            clean_path + img_file, resize=resize, height_resize=resize_ratio)
        improc.save_img(
            processed_path, f"processed_{img_file}", img_configured)


def save_noisy(config):
    noisy_path = config["paths"]["samples"]["noisy"]
    processed_path = config["paths"]["samples"]["processed"]

    processed_imgs = listdir(processed_path)
    for img_file in processed_imgs:
        img_noisy = improc.read_preprocess(
            processed_path + img_file, noise="salt_pepper")
        improc.save_img(noisy_path, f"noisy_{img_file}", img_noisy)


def load_config(path):
    config = dict()
    with open(path, "r") as f:
        config = json.load(f)
    return config


if __name__ == '__main__':
    configs = load_config("../settings.json")
    save_configured(configs)
    save_noisy(configs)
