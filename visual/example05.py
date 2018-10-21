"""!
@brief Example 5
@details: Image filtering examples (smoothing) for noisy image (salt and
pepper noise artificially added)
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
def add_salt_and_pepper(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
filename = '../data/images_general/new_york.jpg'
img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
row, col = img.shape
img2 = add_salt_and_pepper(img, 0.1)
img3 = cv2.medianBlur(img2, 3)
img4 = cv2.GaussianBlur(img2, (3, 3), 0)
plt.subplot(2,2,1); plt.imshow(img, cmap='gray'); plt.title("original")
plt.subplot(2,2,2); plt.imshow(img2, cmap='gray'); plt.title("noisy")
plt.subplot(2,2,3); plt.imshow(img3, cmap='gray'); plt.title("med filtered")
plt.subplot(2,2,4); plt.imshow(img4, cmap='gray'); plt.title("gau filtered")
plt.show()