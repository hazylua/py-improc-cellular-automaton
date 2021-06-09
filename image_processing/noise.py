""" Add noise to grayscale images. """

import numpy as np
from scipy import ndimage


def add_noise(image, noise, rate=0.05):
    """ Adds noise to image. """

    if noise == "gaussian":
        row, col = image.shape
        var = ndimage.laplace(image).var()
        sigma = (var*rate) ** 0.5
        print(var, sigma)
        gauss = np.random.normal(loc=0, scale=sigma, size=(row, col)) * rate
        noisy = image + gauss
        # noisy = image + gauss
        return noisy

    elif noise == "salt_pepper":
        output = image.copy()
        black = 0
        white = 255
        probs = np.random.random(image.shape[:2])
        output[probs < (rate / 2)] = black
        output[probs > 1 - (rate / 2)] = white

        return output

    else:
        return image
