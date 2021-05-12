"""
Image processing helpers.
"""
import random
import cv2 as cv
import numpy as np
from file_operations import check_dir


def add_noise(image, noise, rate=0.005):
    """ Adds noise to image. """

    if noise == "salt_pepper":
        prob = rate
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
    [h, w] = img.shape[:2]
    print(h, w)
    if width is None and height is None:
        return img
    if height is not None and height < h:
        r = height / float(h)
        dim = (int(w * r), height)
        [w, h] = dim
    if width is not None and width < w:
        r = width / float(w)
        dim = (width, int(h * r))

    print(dim, height, width)
    resized = cv.resize(img, dim, interpolation=inter)
    return resized

def read_preprocess(img_path, resize=False, height_resize=None, width_resize=None, noise=False, rate=0.005):
    """ Image pre-processing routine. """

    print('Pre-processing...')
    img = cv.imread(img_path)
    
    if(resize):
        img = image_resize(img, height=height_resize, width=width_resize)

    # Ex.: "salt_pepper"
    if(noise):
        img = add_noise(img, noise, rate=rate)
        
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
