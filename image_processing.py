from file_operations import check_dir, clear_dir

import cv2 as cv
import numpy as np
def image_resize(img, width=None, height=None, inter=cv.INTER_AREA):
    """ Resizes image keeping aspect ratio. """
    if inter == cv.INTER_AREA:
        print('yeyeye')
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


def read_preprocess(img_path):
    print('Pre-processing...')
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print('Image loaded.')
    return img

def save_img(dir_path, file_name, img):
    print('Saving image...')
    if check_dir(dir_path, make_dir=True):
        try:
            cv.imwrite(f'{dir_path + file_name}', np.float32(img))
            print(f'Image saved successfully at: {dir_path + file_name}')
        except Exception as e:
            print(f'Failed to save to {file_name}.\nReason: {e}')
            return False