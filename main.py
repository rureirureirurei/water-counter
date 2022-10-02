import cv2 as cv
import numpy as np
import sys
#import tensorflow
#from keras.models import load_model, Sequential

from modules.config import *
from modules.preprocess import preprocess
from modules.mser import find_digits_potential_locations

def remove_smaller_contours(gray):
    contours,_ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    areas = list(map(lambda x: cv.contourArea(cv.convexHull(x)), contours))
    biggest_contour = contours[np.argmax(areas)]
    mask = np.zeros(np.shape(gray), np.uint8)
    cv.drawContours(mask, [biggest_contour], 0, 255, -1)
    return gray



if __name__ == '__main__':
    image = cv.imread(IMAGE_PATH)
    if image is None:
        print('Could not open or find the image:', IMAGE_PATH)
        exit(0)

    gray = preprocess(image)
    digits, corners, cropped, mser_contours = find_digits_potential_locations(gray)
    digits_threshed = list(map(lambda digit : cv.adaptiveThreshold(digit,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,15,3),digits))
    digits_threshed = np.divide(digits_threshed,255)

    digits_flattened = digits_threshed.reshape(len(digits_threshed),1,784)
    digits_flattened.astype('float32')

    vis = cv.cvtColor(cropped.copy(), cv.COLOR_GRAY2BGR)
    cv.drawContours(vis, mser_contours,-1, (255,0,0),-1)
    cv.imshow('vis',vis)
    for i,digit in enumerate(digits):
        cv.imshow(str(i), digit)
#
#    mnist_model = load_model('keras_mnist.h5')
#    
#    predicted = list(map(lambda a : mnist_model.predict(a), digits_flattened))
#    predicted_classes = np.argmax(predicted,axis=2)
#
#    for i,digit in enumerate(digits_threshed):
#        cv.imshow(str(i)+' prediction: ' + str(predicted_classes[i]), digit)
#        print(predicted[int(predicted_classes[i])], i)
#        print(predicted_classes[i], i)

    cv.waitKey(0)
    cv.destroyAllWindows()
