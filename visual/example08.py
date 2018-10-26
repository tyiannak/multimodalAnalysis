"""!
@brief Example 8
@details: FFT2 Examples  - visual interpretation of FFT2
(log is not used for visualization reasons - the examples need to look quite
"binary")
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

def demo_fft_2(file_path):
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
    fft = (np.abs(fftpack.fftshift(fftpack.fft2(img))))
    h, w = fft.shape
    c1, c2 = int(h / 2), int(w / 2)
    fft_cut = fft[c1 - int(0.025 * h): c1 + int(0.025 * h),
                c2 - int(0.025 * w): c2 + int(0.025 * w)]
    return img, fft, fft_cut
im1, fft1, fft_z1 = demo_fft_2('../data/images_general/sin1.png')
im3, fft3, fft_z2 = demo_fft_2('../data/images_general/sin3.png')
im4, fft4, fft_z4 = demo_fft_2('../data/images_general/sin4.png')
im5, fft5, fft_z5 = demo_fft_2('../data/images_general/sin5.png')
plt.subplot(4, 2, 1); plt.imshow(im1, cmap='gray'); plt.title("image")
plt.axis('off')
plt.subplot(4, 2, 2); plt.imshow(fft_z1, cmap='gray'); plt.title("fft x20 zoom")
plt.axis('off')
plt.subplot(4, 2, 3); plt.imshow(im3, cmap='gray'); plt.title("image")
plt.axis('off')
plt.subplot(4, 2, 4); plt.imshow(fft_z2, cmap='gray'); plt.title("fft x20 zoom")
plt.axis('off')
plt.subplot(4, 2, 5); plt.imshow(im4, cmap='gray'); plt.title("image")
plt.axis('off')
plt.subplot(4, 2, 6); plt.imshow(fft_z4, cmap='gray'); plt.title("fft x20 zoom")
plt.subplot(4, 2, 7); plt.imshow(im5, cmap='gray'); plt.title("image")
plt.axis('off')
plt.subplot(4, 2, 8); plt.imshow(fft_z5, cmap='gray'); plt.title("fft x20 zoom")
plt.axis('off')
plt.show()