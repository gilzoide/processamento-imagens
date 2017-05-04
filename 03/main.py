#!/usr/bin/env python
#
# Name: Gil Barbosa Reis
# Nusp: 8532248
#
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import signal


def main():
    # 1 - Input reading
    signal_name = input()
    s = float(input())
    c = float(input())
    show = input() == '1'
    # 2 - Read the signal
    F = np.fromfile(signal_name, dtype=np.int32)
    # 3 - Apply the Gaussian Windowing
    sigma = len(F) / s
    window = signal.gaussian(len(F), sigma)
    G = F * window
    # 4 - FFT(F)
    # 5 - FFT(G)
    # (Show flag == True) 6 - Plot F, G, FFT(F), FFT(G)
    if show:
        fig = plt.figure()
        a1 = fig.add_subplot(211)
        a2 = fig.add_subplot(212)
        a1.plot(F)
        a2.plot(G)
        plt.show()

if __name__ == '__main__':
    main()

