#!/usr/bin/env python
#
# Name: Gil Barbosa Reis
# Nusp: 8532248
#
# -*- coding: utf-8 -*-

import numpy as np
import cv2

def apply_transformations(image, gamma, a, b):
    return image, image, image, image

def rmsd(original, transformed):
    return ''

def imLog(image):
    pass

def imGamma(image, gamma):
    pass

def imEqualHist(image):
    pass

def imSharp(image, a, b):
    pass

def imHistogram(image):
    pass

def showHistogram(hist):
    pass

def main():
    # 1 - Input reading
    image_name = input()
    gamma = float(input())
    a = float(input())
    b = float(input())
    show_images = input() == '1'
    # 2 - Read the image
    image = cv2.imread(image_name)
    # 3 - Apply transformations
    L, G, H, S = apply_transformations(image, gamma, a, b)
    # (Image show flag == True) 4 - Show all images
    if show_images:
        cv2.imshow('image', image)
        cv2.imshow('logarithm', L)
        cv2.imshow('gamma adjustment', G)
        cv2.imshow('histogram equalisation', H)
        cv2.imshow('sharpening', S)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # (Image show flag == True) 5 - Show the histogram
    if show_images:
        for h in map(imHistogram, [image, L, G, H, S]):
            showHistogram(h)
    # 6 - Output the Root Mean Squared Error
    lines = [
        'RMSD',
        'L=' + rmsd(image, L),
        'G=' + rmsd(image, G),
        'H=' + rmsd(image, H),
        'S=' + rmsd(image, S),
        ''
    ]
    print('\n'.join(lines))

if __name__ == '__main__':
    main()

