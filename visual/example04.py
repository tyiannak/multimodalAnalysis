"""!
@brief Example 4
@details: Image filtering examples (smoothing)
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

filename = '../data/images_general/beach.jpg'
img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
s = 7 # set kernel size
kernel = np.ones((s, s),np.float32)/(s**2)
img2 = cv2.filter2D(img, -1, kernel=kernel)
img3 = cv2.GaussianBlur(img, (7, 7), 0)
img4 = cv2.medianBlur(img, 7)
plt.subplot(2,2,1); plt.imshow(img, cmap='gray'); plt.title("original")
plt.subplot(2,2,2); plt.imshow(img2, cmap='gray'); plt.title("averaging")
plt.subplot(2,2,3); plt.imshow(img3, cmap='gray'); plt.title("gaussian")
plt.subplot(2,2,4); plt.imshow(img4, cmap='gray'); plt.title("median")
plt.show()