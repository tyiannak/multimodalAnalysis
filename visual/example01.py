"""!
@brief Example 1
@details: Load a jpeg image and show it, using opencv, along with the average
values of the RGB coefficients
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import cv2
import matplotlib.pyplot as plt

def img_read_and_plot_rgb_values(filename):
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    print(img.shape)
    print("R range: {} - {} mu={}".format(img[:, :, 0].min(),
                                          img[:, :, 0].max(),
                                          img[:, :, 0].mean()))
    print("G range: {} - {} mu={}".format(img[:, :, 1].min(),
                                          img[:, :, 1].max(),
                                          img[:, :, 1].mean()))
    print("B range: {} - {} mu={}".format(img[:, :, 2].min(),
                                          img[:, :, 2].max(),
                                          img[:, :, 2].mean()))
    plt.imshow(img)
    plt.show()

img_read_and_plot_rgb_values('../data/images_general/beach.jpg')
img_read_and_plot_rgb_values('../data/images_general/new_york.jpg')
