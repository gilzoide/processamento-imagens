#!/usr/bin/env python
#
# Name: Gil Barbosa Reis
# Nusp: 8532248
#
# -*- coding: utf-8 -*-

import numpy as np
import cv2

import random

# pixel generators
FUNCTIONS = [
    lambda x, y, Q: x + y,
    lambda x, y, Q: np.abs(np.sin(x / Q) * 255),
    lambda x, y, Q: ((x / Q) ** 2 + 2 * (y / Q) ** 2) * 255,
    lambda x, y, Q: random.uniform(0, 255),
]

def generate_image(dimension, func_index, Q):
    new_image = np.empty((dimension, dimension), dtype=np.uint8)
    func = FUNCTIONS[func_index]
    # fill in pixels on the image
    for x in range(dimension):
        for y in range(dimension):
            new_image[x][y] = func(x, y, Q)
    return new_image

def main():
    # 1 - Input reading
    image_name = input()
    dimension = int(input())
    func_index = int(input())
    Q = float(input())
    # 2 - Image generation
    image = generate_image(dimension, func_index - 1, Q)
    # 3 - Show image
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 4 - Save image
    cv2.imwrite(image_name, image)

if __name__ == '__main__':
    main()

