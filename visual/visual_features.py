"""!
@brief Visual Features
@details: This script stores the functions required for basic image feature
extraction
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from numpy.lib.stride_tricks import as_strided as ast
import os
import pickle
import scipy
import numpy as np


class ImageFeatureExtractor():
    """@brief  Image Feature Extractor Class
    @param list_of_features (\a list) list of features to extract. One or
    more from []
    """
    def __init__(self, list_of_features=["lbps", "hogs", "colors"]):
        # get all annotations using multiple annotator flag:
        self.list_of_features = list_of_features


    def getRGBS(self, image):
        n_bins_per_hist = 16
        chans = cv2.split(image)
        colors = ("r", "g", "b")
        features = []
        features_sobel = []
        # RGB Histograms:
        for (chan, color) in zip(chans, colors):  # loop over the image channels
            # get histograms
            hist = cv2.calcHist([chan], [0], None, [n_bins_per_hist], [0, 256])
            hist = hist / hist.sum()
            features.extend(hist[:, 0].tolist())
        features.extend(features_sobel)
        # Gray Histogram
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([chan], [0], None, [n_bins_per_hist], [0, 256])
        hist_gray = hist_gray / hist_gray.sum()
        features.extend(hist_gray[:, 0].tolist())
        grad_x = np.abs(
            cv2.Sobel(img_gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0,
                      borderType=cv2.BORDER_DEFAULT))
        grad_y = np.abs(
            cv2.Sobel(img_gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0,
                      borderType=cv2.BORDER_DEFAULT))
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        histSobel = cv2.calcHist([dst], [0], None, [n_bins_per_hist], [0, 256])
        histSobel = histSobel / histSobel.sum()
        features.extend(histSobel[:, 0].tolist())
        # HSV histogram
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        chans = cv2.split(hsv)  # take the S channel
        S = chans[1]
        hist2 = cv2.calcHist([S], [0], None, [n_bins_per_hist], [0, 256])
        hist2 = hist2 / hist2.sum()
        features.extend(hist2[:, 0].tolist())
        f_names = ["Color-R" + str(i) for i in range(n_bins_per_hist)]
        f_names.extend(["Color-G" + str(i) for i in range(n_bins_per_hist)])
        f_names.extend(["Color-B" + str(i) for i in range(n_bins_per_hist)])
        f_names.extend(["Color-Gray" + str(i) for i in range(n_bins_per_hist)])
        f_names.extend(["Color-GraySobel" + str(i) for i in range(n_bins_per_hist)])
        f_names.extend(["Color-Satur" + str(i) for i in range(n_bins_per_hist)])
        return np.array(features), f_names

    def block_view(self, A, block=(32, 32)):
        shape = (A.shape[0] // block[0], A.shape[1] // block[1]) + block
        strides = (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides

        return ast(A, shape=shape, strides=strides)

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def getLBP(self, img):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        radius = 1
        n_points = 8 * radius
        lbpImage = (local_binary_pattern(img2, n_points, radius)).astype(
            int) ** (1.0 / radius)
        # block processing:
        lbpImages = self.block_view(lbpImage,
                                    (int(lbpImage.shape[0] / 4),
                                     int(lbpImage.shape[1] / 4)))
        count = 0
        LBP = np.array([]);
        for i in range(lbpImages.shape[0]):  # for each block:
            for j in range(lbpImages.shape[1]):
                count += 1
                LBPt = cv2.calcHist([lbpImages[i, j, :, :].astype('uint8')],
                                    [0], None, [8], [0, 256])
                LBP = np.append(LBP, LBPt[:, 0]);
        f_names = ["LBP" + str(i).zfill(2) for i in range(len(LBP))]
        return self.normalize(LBP), f_names

    def getHOG(self, img, n_hogs_per_dim = 4):
        pixels_per_cell_h = img.shape[0] // n_hogs_per_dim
        pixels_per_cell_w = img.shape[1] // n_hogs_per_dim
        fd = hog(img, orientations=8,
                  pixels_per_cell=(pixels_per_cell_h, pixels_per_cell_w),
                  cells_per_block=(1, 1), visualize=False, multichannel=True)
        f_names = ['HOG' + str(i).zfill(2) for i in range(len(fd))]
        return fd, f_names

    def getKAZE(self, image, vector_size=8):
        alg = cv2.KAZE_create()
        # Finding image keypoints
        kps = alg.detect(image)
        # Getting first vector_size of them
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        fd = dsc.flatten()
        # Making descriptor of same size  (descriptor vector size is 64)
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            fd = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        f_names = ['KAZE' + str(i).zfill(2) for i in range(len(fd))]
        return fd, f_names

    def extract_features(self, file_path):
        # read image
        img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        features = []
        feature_names = []
        if "colors" in self.list_of_features:
            f, fn = self.getRGBS(img)
            features.append(f)
            feature_names += fn
        if "lbps" in self.list_of_features:
            f, fn = self.getLBP(img)
            features.append(f)
            feature_names += fn
        if "hogs" in self.list_of_features:
            f, fn = self.getHOG(img)
            features.append(f)
            feature_names += fn
        if "kaze" in self.list_of_features:
            f, fn = self.getKAZE(img)
            features.append(f)
            feature_names += fn
        features = np.concatenate(features)
        return features, feature_names

class ImageMatch(object):
    def __init__(self, pickled_db_path="features.pck"):
        with open(pickled_db_path, 'rb') as fp:
            self.data = pickle.load(fp)
        self.names, self.matrix = [], []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix, self.names = np.array(self.matrix), np.array(self.names)

    def cos_cdist(self, vector): # cosine distance between search image and db
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    def match(self, image_path, topn=5):
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        ife = ImageFeatureExtractor()
        features, _ = ife.getKAZE(img, 32)
        im_dist = self.cos_cdist(features)
        nearest_i = np.argsort(im_dist)[:topn].tolist()  # top 5 records
        nearest_paths = self.names[nearest_i].tolist()
        return nearest_paths, im_dist[nearest_i].tolist()

def batch_extractor_match(images_path, pickled_db_path="features.pck"):
    files = [os.path.join(images_path, p) for p in
             sorted(os.listdir(images_path))]
    result = {}
    for f in files:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        ife = ImageFeatureExtractor()
        result[f.split('/')[-1].lower()], _ = ife.getKAZE(img, 32)
    with open(pickled_db_path, 'wb') as fp:  # save feature vectors in pickle
        pickle.dump(result, fp)
