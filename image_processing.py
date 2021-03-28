import file_operations as fileops

import cv2 as cv

def read_preprocess(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img