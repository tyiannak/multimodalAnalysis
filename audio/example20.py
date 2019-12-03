"""! 
@brief Example 20
@details Musical genre classification example. Classification performance
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np
import utilities as ut
from pyAudioAnalysis.MidTermFeatures import directory_feature_extraction as dW

if __name__ == '__main__':
    # extract features, concatenate feature matrices and normalize:
    mw, stw = 2, .1
    f1, _, fn1 = dW("../data/musical_genres_8k/blues", mw, mw, stw, stw)
    f2, _, fn2 = dW("../data/musical_genres_8k/electronic", mw, mw, stw, stw)
    f3, _, fn3 = dW("../data/musical_genres_8k/jazz", mw, mw, stw, stw)
    x = np.concatenate((f1, f2, f3), axis=0)
    y = np.concatenate((np.zeros(f1.shape[0]), np.ones(f2.shape[0]),
                       2 * np.ones(f2.shape[0])))
    # train svm and get aggregated (average) confusion matrix, accuracy and f1
    cm, acc, f1 = ut.svm_train_evaluate(x, y, 10, C=2)
    # visualize performance measures
    ut.plotly_classification_results(cm, ["blues", "electronic", "jazz"])
    print(acc, f1)

