import cv2 as cv
import numpy as np
import sys
from modules.config import *

def _filter_regions_corners(regions):
    filtered_regions = []

    for index in range(len(regions) - 1):
        cur = regions[index]
        nxt = regions[index + 1]
        delta = 0
        for crd in range(0, 4):
            delta += abs(cur[crd] - nxt[crd])
        if (delta < DELTA_THRESHOLD):
            continue
        filtered_regions.append(cur)

    filtered_regions.append(regions[-1])

    return filtered_regions


def _mser(img):
    mser = cv.MSER_create(MSER_DELTA, MSER_MIN_AREA, MSER_MAX_AREA)
    regions, _ = mser.detectRegions(img)
    corners = []

    for region in regions:
        xmax, ymax = np.amax(region, axis=0)
        xmin, ymin = np.amin(region, axis=0)
        corners.append((xmin, ymin, xmax, ymax))

    corners = _filter_regions_corners(corners)

    return corners, regions

def find_digits_potential_locations(gray):
    ret,thresh = cv.threshold(gray,175,255,cv.THRESH_BINARY)

    contours,_ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    mask = gray.copy()
    mask = cv.bitwise_and(mask, 0)

    boundingRect = []
    for contour in contours:
            if cv.contourArea(contour) >= 8000:
                cv.drawContours(mask, [contour], 0, 255, -1)
                boundingRect = cv.boundingRect(contour)

    xmin = boundingRect[0]
    ymin = boundingRect[1]
    xmax = boundingRect[2] + boundingRect[0]
    ymax = boundingRect[3] + boundingRect[1]

    gray = cv.bitwise_and(gray,mask)
    gray_cropped = gray[ymin:ymax,xmin:xmax]
    gray_cropped_resized = cv.resize(gray_cropped, (CROPPED_WIDTH,CROPPED_HEIGHT))
    gray = gray_cropped_resized.copy()

    corners, regions = _mser(gray)

    digit_potential_images = []
    for corner in corners:
        xmin = corner[0]
        ymin = corner[1]
        xmax = corner[2]
        ymax = corner[3]
        digit_image = gray[ymin:ymax,xmin:xmax]
        digit_image = cv.resize(digit_image, (DIGIT_WIDTH, DIGIT_HEIGHT))
        digit_potential_images.append(digit_image)

    return (digit_potential_images, corners, gray_cropped_resized, regions)


