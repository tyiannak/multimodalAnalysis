"""!
@brief Example 10
@details: FFT2 and kernel filtering relationship (show freq responses
of basic filters)
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2


mean_filter = np.ones((5, 5)) #  averaging filter

x = cv2.getGaussianKernel(5,10)
gaussian = x*x.T # gaussian filter

# sobel in x direction
sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
# sobel in y direction
sobel_y= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])

filters = [mean_filter, gaussian, sobel_x, sobel_y]
filter_name = ['mean_filter', 'gaussian', 'sobel_x', 'sobel_y']
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]
for i in range(4):
    plt.subplot(2, 2, i+1),plt.imshow(mag_spectrum[i], cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
plt.show()