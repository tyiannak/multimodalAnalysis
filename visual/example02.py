"""!
@brief Example 2
@details: Load 2 jpeg images and show them, using opencv, along with the
average HSV coefficients
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import cv2
import matplotlib.pyplot as plt

def img_read_and_plot_hsv_values(filename):
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2HSV)
    print(img.shape)
    print("H range: {} - {} mu={}".format(img[:, :, 0].min(),
                                          img[:, :, 0].max(),
                                          img[:, :, 0].mean()))
    print("S range: {} - {} mu={}".format(img[:, :, 1].min(),
                                          img[:, :, 1].max(),
                                          img[:, :, 1].mean()))
    print("V range: {} - {} mu={}".format(img[:, :, 2].min(),
                                          img[:, :, 2].max(),
                                          img[:, :, 2].mean()))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
    plt.show()

img_read_and_plot_hsv_values('../data/images_general/beach.jpg')
img_read_and_plot_hsv_values('../data/images_general/new_york.jpg')
