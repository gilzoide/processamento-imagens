#!/usr/bin/env python
#
# Name: Gil Barbosa Reis
# Nusp: 8532248
#
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt

def _flt2uchar(imgf):
    '''Auxiliary function to transform back a float32 img to a uchar one'''
    aux = cv2.convertScaleAbs(imgf)
    return cv2.normalize(aux, aux, 0, 255, cv2.NORM_MINMAX)

def apply_transformations(image, gamma, a, b):
    L = imLog(image)
    G = imGamma(image, gamma)
    H = imEqualHist(image)
    S = imSharp(image, a, b)
    return L, G, H, S

def rmsd(original, transformed):
    return np.sqrt(np.mean((original - transformed) ** 2))

def imLog(image):
    _, maximg, _, _ = cv2.minMaxLoc(image)
    imgf = np.float32(image)
    imgf = cv2.log(imgf + 1)
    imgf = cv2.multiply(imgf, (255 / np.log(1 + maximg)))
    return _flt2uchar(imgf)

def imGamma(image, gamma):
    imgf = np.float32(image)
    new_img = np.power(imgf, gamma)
    return cv2.convertScaleAbs(new_img)

def imEqualHist(image):
    hist = imHistogram(image)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf[image]

sharp_kernel = np.array([
    .05, .1, .05,
    .1, .4, .1,
    .05, .1, .05,
])
def imSharp(image, a, b):
    B = cv2.filter2D(image, -1, sharp_kernel)
    lhs = cv2.multiply(image, a)
    rhs = cv2.multiply(cv2.subtract(B, image), b)
    return _flt2uchar(cv2.add(lhs, rhs))

def imHistogram(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    return hist

def showHistogram(hist, name):
    plt.figure(name)
    plt.plot(hist)
    plt.show()

def main():
    # 1 - Input reading
    image_name = input()
    gamma = float(input())
    a = float(input())
    b = float(input())
    show_images = input() == '1'
    # 2 - Read the image
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    # 3 - Apply transformations
    L, G, H, S = apply_transformations(image, gamma, a, b)

    image_list = [image, L, G, H, S]
    image_names = ['image', 'logarithm', 'gamma adjustment', 'histogram equalisation', 'sharpening']
    # (Image show flag == True) 4 - Show all images
    if show_images:
        for img, name in zip(image_list, image_names):
            cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # (Image show flag == True) 5 - Show the histogram
    if show_images:
        for h, name in zip(map(imHistogram, image_list), image_names):
            showHistogram(h, name)
    # 6 - Output the Root Mean Squared Error
    lines = [
        'RMSD',
        'L=' + str(rmsd(image, L)),
        'G=' + str(rmsd(image, G)),
        'H=' + str(rmsd(image, H)),
        'S=' + str(rmsd(image, S)),
        ''
    ]
    print('\n'.join(lines))

if __name__ == '__main__':
    main()

