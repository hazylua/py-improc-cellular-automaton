import numpy as np
import cv2 as cv
from file_processing import check_dir

from .noise import add_noise
from .transformation import image_resize


def read_preprocess(img_path, resize=False, height_resize=None, width_resize=None, noise=False, rate=0.005):
    """ Image pre-processing routine. """

    print('Pre-processing...')
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    if(resize):
        img = image_resize(img, height=height_resize, width=width_resize)

    # Ex.: "salt_pepper"
    if(noise):
        img = add_noise(img, noise, rate=rate)
        

    print('Image loaded.')
    return img


def save_img(dir_path, file_name, img):
    """ Saves image and checks for save location. """

    print('Saving image...')
    if check_dir(dir_path, make_dir=True):
        try:
            cv.imwrite(f'{dir_path + file_name}', np.float32(img))
            print(f'Image saved successfully at: {dir_path + file_name}')
        except Exception as e:
            print(f'Failed to save to {file_name}.\nReason: {e}')
            return False
