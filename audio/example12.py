"""! 
@brief Example 11
@details pyAudioAnalysis chromagram example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
from pyAudioAnalysis import audioFeatureExtraction as aF
import os.path
import utilities as ut

if __name__ == '__main__':
    dirs = ["../data/gender/male",
            "../data/gender/female"]
    class_names = [os.path.basename(d) for d in dirs]
    m_win, m_step, s_win, s_step = 1, 1, 0.1, 0.05
    features = []
    for d in dirs:
        # get feature matrix for each directory (class)
        f, files, fn = aF.dirWavFeatureExtraction(d, m_win, m_step, s_win,
                                                  s_step)
        features.append(f)
    ut.plot_feature_histograms(features, fn, class_names)