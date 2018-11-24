"""!
@brief Example 12
@details: Visual features visualization for 2 images using matplotlib.
(Color features)
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

from visual_features import ImageFeatureExtractor
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt

filename1 = '../data/images_general/beach.jpg'
filename2 = '../data/images_general/new_york.jpg'

ife = ImageFeatureExtractor(list_of_features=["colors"])
f1, f1n = ife.extract_features(file_path=filename1)
f2, f2n = ife.extract_features(file_path=filename2)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(f1n)), f1)
ax.set_xlim(xmin=0)
plt.xticks(range(len(f1n)))
ax.set_xticklabels(f1n)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.setp(plt.xticks()[1], rotation=90)
plt.tick_params(axis='both', which='major', labelsize=6)
plt.tick_params(axis='both', which='minor', labelsize=6)
plt.show()
