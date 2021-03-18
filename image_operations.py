import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_path = './white_cat.jpg'

def bitmap():
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    binary = cv.threshold(img,127,255,cv.THRESH_BINARY)
    return binary

def save(img, path):
    cv.imwrite(path, img)