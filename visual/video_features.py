"""!
@brief Visual Features
@details: This script stores the functions required for basic video feature
extraction
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import cv2
import visual_features
import os
import pickle
import scipy
import numpy as np
import tqdm


class VideoFeatureExtractor():
    """@brief  Video Feature Extractor Class
    @param list_of_features (\a list) list of features to extract. One or
    more from []
    @param resize_width (\a int) width of resized image to be used before
    feature extraction (-1 for not resize)
    """
    def __init__(self, list_of_features=["lbps", "hogs", "colors"],
                 resize_width=-1, step=0.1):
        # get all annotations using multiple annotator flag:
        self.list_of_features = list_of_features
        self.resize_width = resize_width
        self.step = step

    def resize_image(self, img, target_width):
        (width, height) = img.shape[1], img.shape[0]
        if target_width != -1:  # Use target_width = -1 for NO frame resizing
            ratio = float(width) / target_width
            new_h = int(round(float(height) / ratio))
            img_new = cv2.resize(img, (target_width, new_h))
        else:
            img_new = img
        return img_new

    def extract_features(self, file_path):
        # read video
        capture = cv2.VideoCapture(file_path)
        nFrames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        duration = nFrames / fps
        nextTimeStampToProcess = 0.0
        count = 0
        vs = visual_features.ImageFeatureExtractor()
        frames_to_process = int(duration/self.step)
        print(frames_to_process)
        pbar = tqdm.tqdm(total=frames_to_process)
        features_all = []
        timestamps = []
        while 1:
            ret, frame = capture.read()
            timeStamp = float(count) / fps
            if timeStamp >= nextTimeStampToProcess:
                nextTimeStampToProcess += self.step;
                PROCESS_NOW = True
            if ret:
                count += 1
                if PROCESS_NOW:
                    timestamps.append(timeStamp)
                    pbar.update(1)
                    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame3 = self.resize_image(frame2, self.resize_width)
                    features = []
                    feature_names = []
                    cv2.imshow('Color', cv2.cvtColor(frame3, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(5) & 0xFF

                    if "colors" in self.list_of_features:
                        f, fn = vs.getRGBS(frame3)
                        features.append(f)
                        feature_names += fn
                    if "lbps" in self.list_of_features:
                        f, fn = vs.getLBP(frame3)
                        features.append(f)
                        feature_names += fn
                    if "hogs" in self.list_of_features:
                        f, fn = vs.getHOG(frame3)
                        features.append(f)
                        feature_names += fn
                    if "kaze" in self.list_of_features:
                        f, fn = vs.getKAZE(frame3)
                        features.append(f)
                        feature_names += fn
                    features = np.concatenate(features)
                    features_all.append(features)
                    PROCESS_NOW = False
            else:
                break
        return np.array(features_all), np.array(timestamps), feature_names

