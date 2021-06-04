import random
import numpy as np

def add_noise(image, noise, rate=0.005):
    """ Adds noise to image. """

    if noise == "gaussian":
        row, col = image.shape
        mean = 0
        var = col
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        # gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy

    elif noise == "salt_pepper":
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