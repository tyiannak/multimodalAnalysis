"""!
@brief Example 6
@details: Edge detection (canny)
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import cv2
import matplotlib.pyplot as plt

filename = '../data/images_general/pyramid.jpg'
img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
img_2 = cv2.Canny(img, 150, 220)
img_3 = cv2.medianBlur(img, 7)
img_4 = cv2.Canny(img_3, 150, 220)
plt.subplot(2, 2, 1); plt.imshow(img, cmap='gray'); plt.title("original")
plt.subplot(2, 2, 2); plt.imshow(img_2, cmap='gray'); plt.title("canny")
plt.subplot(2, 2, 3); plt.imshow(img_3, cmap='gray'); plt.title("medfilter")
plt.subplot(2, 2, 4); plt.imshow(img_4, cmap='gray'); plt.title("canny")
plt.show()