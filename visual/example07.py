"""!
@brief Example 7
@details: FFT2 Examples
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

img1 = cv2.cvtColor(cv2.imread('../data/images_general/pyramid.jpg'),
                    cv2.COLOR_BGR2GRAY)
fft1 = np.log(np.abs(fftpack.fftshift(fftpack.fft2(img1))))

img2 = cv2.cvtColor(cv2.imread('../data/images_general/new_york_1.jpg'),
                    cv2.COLOR_BGR2GRAY)
fft2 = np.log(np.abs(fftpack.fftshift(fftpack.fft2(img2))))

plt.subplot(2, 2, 1); plt.imshow(img1, cmap='gray'); plt.title("image")
plt.subplot(2, 2, 2); plt.imshow(fft1, cmap='jet'); plt.title("fft")
plt.subplot(2, 2, 3); plt.imshow(img2, cmap='gray'); plt.title("image")
plt.subplot(2, 2, 4); plt.imshow(fft2, cmap='jet'); plt.title("fft")
plt.show()