"""!
@brief Visual Features
@details: This script stores the functions required for basic video feature
extraction
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import cv2
import visual_features
import numpy as np
import tqdm

def angleDiff(angle1, angle2):
    # angles (in degs) difference
    Diff = np.abs(angle2-angle1);
    if np.abs(Diff) > 180:
        Diff = 360 - Diff
    return Diff

def anglesSTD(angles, MEAN):
    # computes the standard deviation between a set of angles
    S = 0.0;
    for a in angles:
        S += (angleDiff(a, MEAN)**2)
    S /= len(angles)
    S = np.sqrt(S)
    return S

class VideoFeatureExtractor():
    """@brief  Video Feature Extractor Class
    @param list_of_features (\a list) list of features to extract. One or
    more from []
    @param resize_width (\a int) width of resized image to be used before
    feature extraction (-1 for not resize)
    """
    def __init__(self, list_of_features=["lbps", "hogs", "colors","flow"],
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

    def computeFlowFeatures(self, Grayscale, GrayscalePrev, p0, lk_params):
        p1, st, err = cv2.calcOpticalFlowPyrLK(GrayscalePrev, Grayscale, p0,
                                               None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        angles = []
        mags = []
        dxS = [];
        dyS = []
        for i, (new, old) in enumerate(
                zip(good_new, good_old)):  # draw motion arrows
            x1, y1 = new.ravel();
            x2, y2 = old.ravel()
            dx = x2 - x1;
            dy = y2 - y1
            if dy < 0:
                angles.append(
                    [np.abs(180.0 * np.arctan2(dy, dx) / np.pi)])
            else:
                angles.append(
                    [360.0 - 180.0 * np.arctan2(dy, dx) / np.pi])
            mags.append(np.sqrt(dx * dx + dy * dy) / np.sqrt(
                Grayscale.shape[0] * Grayscale.shape[0] + Grayscale.shape[1] *
                Grayscale.shape[1]))
            dxS.append(dx);
            dyS.append(dy);
        angles = np.array(angles);
        mags = np.array(mags);
        DistHorizontal = -1;
        if len(angles) > 0:
            meanDx = np.mean(dxS);
            meanDy = np.mean(dyS);
            if meanDy < 0:
                MEANANGLE = -(
                180.0 * np.arctan2(int(meanDy), int(meanDx)) / np.pi);
            else:
                MEANANGLE = 360.0 - (
                180.0 * np.arctan2(int(meanDy), int(meanDx)) / np.pi);
            STD = anglesSTD(angles, MEANANGLE)

            DistHorizontal = min(angleDiff(MEANANGLE, 180),
                                 angleDiff(MEANANGLE, 0))
            TitlPanConfidence = np.mean(mags) / np.sqrt(STD + 0.000000010);
            TitlPanConfidence = TitlPanConfidence[0]
            # TODO:
            # CHECK PANCONFIDENCE
            # SAME FOR ZOOM AND OTHER CAMERA EFFECTS
            if TitlPanConfidence < 1.0:
                TitlPanConfidence = 0;
                DistHorizontal = -1;
        else:
            mags = [0];
            angles = [0];
            dxS = [0];
            dyS = [0];
            MEANANGLE = 0
            STD = 0
            TitlPanConfidence = 0.0
        return (angles, mags, MEANANGLE, STD, good_new, good_old, dxS, dyS,
                TitlPanConfidence)

    def extract_features(self, file_path):
        # read video
        capture = cv2.VideoCapture(file_path)
        n_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        duration, next_timestamp_proc = n_frames / fps, 0.0
        vs = visual_features.ImageFeatureExtractor()
        frames_to_process = int(duration/self.step)
        pbar = tqdm.tqdm(total=frames_to_process)
        features_all, timestamps, count = [], [], 0
        while 1:
            ret, frame = capture.read()
            timestamp = float(count) / fps
            if timestamp >= next_timestamp_proc:
                next_timestamp_proc += self.step;
                PROCESS_NOW = True
            if ret:
                count += 1
                if PROCESS_NOW:
                    timestamps.append(timestamp)
                    pbar.update(1)
                    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame3 = self.resize_image(frame2, self.resize_width)
                    features, feature_names = [], []
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
                    if "flow" in self.list_of_features:
                        lk_params = dict(winSize=(15, 15),
                                         maxLevel=5,
                                         criteria=(
                                         cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                         10, 0.03))
                        feature_params = dict(maxCorners=500, qualityLevel=0.3,
                                              minDistance=3, blockSize=5)
                        frame3_g = cv2.cvtColor(frame3, cv2.COLOR_RGB2GRAY)
                        if count == 1:
                            frame3_g_prev = frame3_g
                        p0 = cv2.goodFeaturesToTrack(frame3_g, mask=None,
                                                     **feature_params)
                        angles, mags, MEANANGLE, STD, good_new, good_old, dxS, dyS, TitlPanConfidence = \
                            self.computeFlowFeatures(frame3_g, frame3_g_prev, p0, lk_params)
                        f = [MEANANGLE, STD]
                        fn = ["m", "s"]
                        features.append(f)
                        feature_names += fn
                        frame3_g_prev = frame3_g
                    features = np.concatenate(features)
                    features_all.append(features)
                    PROCESS_NOW = False
            else:
                break
        return np.array(features_all), np.array(timestamps), feature_names