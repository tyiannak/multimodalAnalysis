"""!
@brief Example 14
@details: Image matching example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import visual_features as vf
from visual_features import ImageMatch
import os, cv2,  matplotlib.pyplot as plt, random

def show_images(img_paths):
    imgs = [cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in img_paths]
    for i, img in enumerate(imgs):
        plt.subplot(2, 2, i+1); plt.imshow(img)
    plt.show()

images_path = "../data/keys"
files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
sample = random.sample(files, 15)  # get 3 random images
vf.batch_extractor(images_path, pickled_db_path="features.pck")
ma = ImageMatch('features.pck')
for s in sample:
    img_paths = [s]
    names, match = ma.match(s, topn=4)
    for n in names[1:]:
        img_paths.append(os.path.join(images_path, n))
    show_images(img_paths)
