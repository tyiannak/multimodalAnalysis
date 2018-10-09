"""! 
@brief Example 24
@details Soundscape quality classification (through svm classifier)
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np
from pyAudioAnalysis.audioFeatureExtraction import dirWavFeatureExtraction as dW
import utilities as ut

if __name__ == '__main__':
    # get features from folders (all classes):
    f1, _, fn1 = dW("../data/soundScape_small/1/", 2, 1, 0.1, 0.1)
    f3, _, fn1 = dW("../data/soundScape_small/3/", 2, 1, 0.1, 0.1)
    f5, _, fn1 = dW("../data/soundScape_small/5/", 2, 1, 0.1, 0.1)

    x = np.concatenate((f1, f3, f5), axis=0)
    y = np.concatenate((np.zeros(f1.shape[0]), 1 * np.ones(f3.shape[0]),
                       2 * np.ones(f5.shape[0])))
    # train svm and get aggregated (average) confusion matrix, accuracy and f1
    cm, acc, f1 = ut.svm_train_evaluate(x, y, 10, C=10, use_regressor=False)
    # visualize performance measures
    ut.plotly_classification_results(cm, ["q_1",  "q_3", "q_5"])
    print(acc, f1)
