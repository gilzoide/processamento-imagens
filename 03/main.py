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
from scipy.fftpack import fft, ifft

def NFS(s):
    return np.absolute(s) / (2.0 * len(s))

def loPass(c, Ffft, Gfft):
    T = c * np.argmax(np.absolute(Gfft))
    return np.array([a if i < T else 0 for i, a in enumerate(Ffft)])

def main():
    # 1 - Input reading
    signal_name = input()
    s = float(input())
    c = float(input())
    show = input().startswith('1')
    # 2 - Read the signal
    F = np.fromfile(signal_name, dtype=np.int32)
    # 3 - Apply the Gaussian Windowing
    sigma = len(F) / s
    window = signal.gaussian(len(F), sigma)
    G = F * window
    # 4 - FFT(F)
    Ffft = fft(F)
    # 5 - FFT(G)
    Gfft = fft(G)
    # (Show flag == True) 6 - Plot F, G, FFT(F), FFT(G)
    if show:
        fig = plt.figure()
        a1 = fig.add_subplot(221)
        a2 = fig.add_subplot(222)
        a3 = fig.add_subplot(223)
        a4 = fig.add_subplot(224)
        a1.plot(F)
        a2.plot(NFS(Ffft)[:len(F) // 2])
        a3.plot(G)
        a4.plot(NFS(Gfft)[:len(G) // 2])
        plt.show()
    # 7 - Low pass filter
    filteredFfft = loPass(c, Ffft, Gfft)
    # 8 - IFFT(filteredF)
    filteredF = np.absolute(ifft(filteredFfft))
    # (Show flag == True) 9 - Plot F, filteredF
    if show:
        fig = plt.figure()
        a1 = fig.add_subplot(211)
        a2 = fig.add_subplot(212)
        a1.plot(F)
        a2.plot(filteredF)
        plt.show()
    # 10 - outputs
    print(np.argmax(np.absolute(Ffft)))
    print(np.argmax(np.absolute(Gfft)))
    print(np.amax(F))
    print(np.amax(filteredF))

if __name__ == '__main__':
    main()

