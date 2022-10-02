import cv2 as cv
import numpy as np
import sys
from modules.config import *

def preprocess(image):
    shrinked = cv.resize(image, (SHRINKED_WIDTH, SHRINKED_HEIGHT))
    gray = cv.cvtColor(shrinked, cv.COLOR_BGR2GRAY)
    return gray
