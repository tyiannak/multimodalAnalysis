"""!
@brief Example 9
@details: FFT2 filtering example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

filename = '../data/images_general/new_york_1.jpg'
img_1 = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
fft_1 = (fftpack.fft2(img_1))
keep_fraction = 0.1
fft_2 = fft_1.copy()
r, c = fft_2.shape
fft_2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
fft_2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
img_2 = np.fft.ifft2(fft_2).real

plt.subplot(2, 2, 1); plt.imshow(img_1, cmap='gray'); plt.title("image")
plt.subplot(2, 2, 2); plt.imshow(np.log(fftpack.fftshift(np.abs(fft_1))),
                                 cmap='jet');
plt.title("fft mag"); plt.axis('off')
plt.subplot(2, 2, 3); plt.imshow(img_2, cmap='gray');
plt.title("filtered image")
plt.subplot(2, 2, 4); plt.imshow(np.log(fftpack.fftshift(np.abs(fft_2))),
                                 cmap='jet');
plt.title("fft filtered mag"); plt.axis('off')
plt.show()