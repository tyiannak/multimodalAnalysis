"""! 
@brief Example 23
@details Audio event detection. Classification performance
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np
import utilities as ut
from pyAudioAnalysis.audioFeatureExtraction import dirWavFeatureExtraction as dW


if __name__ == '__main__':
    # extract features, concatenate feature matrices and normalize:
    f1, _, fn1 = dW("../data/activity_sounds/cupboards", 1, 1, 0.05, 0.05)
    f2, _, fn1 = dW("../data/activity_sounds/door", 1, 1, 0.05, 0.05)
    f3, _, fn1 = dW("../data/activity_sounds/silence", 1, 1, 0.05, 0.05)
    f4, _, fn1 = dW("../data/activity_sounds/walk", 1, 1, 0.05, 0.05)
    x = np.concatenate((f1, f2, f3, f4), axis=0)
    y = np.concatenate((np.zeros(f1.shape[0]), np.ones(f2.shape[0]),
                       2 * np.ones(f3.shape[0]), 3 * np.ones(f4.shape[0])))
    print(x.shape, y.shape)
    # train svm and get aggregated (average) confusion matrix, accuracy and f1
    cm, acc, f1 = ut.svm_train_evaluate(x, y, 5, C=2)
    # visualize performance measures
    ut.plotly_classification_results(cm, ["cupboards", "door", "silence",
                                          "walk"])
    print(acc, f1)

