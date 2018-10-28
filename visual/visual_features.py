"""!
@brief Visual Features
@details: This script stores the functions required for basic image feature
extraction
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from numpy.lib.stride_tricks import as_strided as ast

class ImageFeatureExtractor():
    """@brief  Image Feature Extractor Class
    @param list_of_features (\a list) list of features to extract. One or
    more from []
    @param resize_width (\a int) width of resized image to be used before
    feature extraction (-1 for not resize)
    """
    def __init__(self, list_of_features=["lbps", "hogs"], resize_width=-1):
        # get all annotations using multiple annotator flag:
        self.list_of_features = list_of_features
        self.resize_width = resize_width

    def resize_image(img, target_width):
        (width, height) = img.shape[1], img.shape[0]
        if target_width != -1:  # Use target_width = -1 for NO frame resizing
            ratio = float(width) / target_width
            new_h = int(round(float(height) / ratio))
            img_new = cv2.resize(img, (target_width, new_h))
        else:
            img_new = img
        return img_new

    def getRGBS(self, image):
        chans = cv2.split(image)
        colors = ("r", "g", "b")
        features = []
        features_sobel = []
        # RGB Histograms:
        for (chan, color) in zip(chans, colors):  # loop over the image channels
            # get histograms
            hist = cv2.calcHist([chan], [0], None, [8], [0, 256])
            hist = hist / hist.sum()
            features.extend(hist[:, 0].tolist())
        features.extend(features_sobel)
        # Gray Histogram
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([chan], [0], None, [8], [0, 256])
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
        histSobel = cv2.calcHist([dst], [0], None, [8], [0, 256])
        histSobel = histSobel / histSobel.sum()
        features.extend(histSobel[:, 0].tolist())
        # HSV histogram
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        chans = cv2.split(hsv)  # take the S channel
        S = chans[1]
        hist2 = cv2.calcHist([S], [0], None, [8], [0, 256])
        hist2 = hist2 / hist2.sum()
        features.extend(hist2[:, 0].tolist())
        Fnames = ["Color-R" + str(i) for i in range(8)]
        Fnames.extend(["Color-G" + str(i) for i in range(8)])
        Fnames.extend(["Color-B" + str(i) for i in range(8)])
        Fnames.extend(["Color-Gray" + str(i) for i in range(8)])
        Fnames.extend(["Color-GraySobel" + str(i) for i in range(8)])
        Fnames.extend(["Color-Satur" + str(i) for i in range(8)])
        return features, Fnames

    def block_view(self, A, block=(32, 32)):
        shape = (A.shape[0] / block[0], A.shape[1] / block[1]) + block
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
                                    (int(lbpImage.shape[0] / 2),
                                     int(lbpImage.shape[1] / 4)))
        count = 0
        LBP = np.array([]);
        for i in range(lbpImages.shape[0]):  # for each block:
            for j in range(lbpImages.shape[1]):
                count += 1
                LBPt = cv2.calcHist([lbpImages[i, j, :, :].astype('uint8')],
                                    [0], None, [8], [0, 256])
                LBP = np.append(LBP, LBPt[:, 0]);
        Fnames = ["LBP" + str(i).zfill(2) for i in range(len(LBP))]
        return self.normalize(LBP).tolist(), Fnames


    def extract_features(self, file_path):
        # read image
        img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        f_rgb, f_rgb_names = self.getRGBS(img)
        f_lbps, f_lbps_names = self.getLBP(img)
        f = f_rgb + f_lbps
        f_names = f_rgb_names + f+f_lbps_names
        print f, f_names