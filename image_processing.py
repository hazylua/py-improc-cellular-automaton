"""
Image processing helpers.
"""
import random
import cv2 as cv
import numpy as np
from file_operations import check_dir


def add_noise(image, noise):
    """ Adds noise to image. """

    if noise == "salt_pepper":
        prob = 0.005
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    else:
        return image


def image_resize(img, width=None, height=None, inter=cv.INTER_AREA):
    """ Resizes image keeping aspect ratio. """
    
    dim = None
    (h, w) = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:

        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv.resize(img, dim, interpolation=inter)
    return resized


def read_preprocess(img_path, resize=False, height_resize=None, width_resize=None):
    """ Image pre-processing routine. """

    print('Pre-processing...')
    img = cv.imread(img_path)
    
    if(resize):
        h = None
        w = None
        if(height_resize):
            h = int(img.shape[1] * height_resize)
        elif(width_resize):
            w = int(img.shape[0] * width_resize)
        img = image_resize(img, height=h, width=w)

    img = add_noise(img, "salt_pepper")
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

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
