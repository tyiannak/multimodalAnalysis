"""! 
@brief Example 20
@details Musical genre classification example. Classification performance
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np, plotly.graph_objs as go
from pyAudioAnalysis.audioFeatureExtraction import dirWavFeatureExtraction as dW
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

if __name__ == '__main__':
    f1, _, fn1 = dW("../data/musical_genres_8k/blues", 2, 1, 0.1, 0.1)
    f2, _, fn2 = dW("../data/musical_genres_8k/electronic", 2, 1, 0.1, 0.1)
    f3, _, fn3 = dW("../data/musical_genres_8k/jazz", 2, 1, 0.1, 0.1)
    x = np.concatenate((f1, f2, f3), axis=0)
    mean, std = x.mean(axis=0), np.std(x, axis=0)
    x = (x - mean) / std

    y = np.concatenate((np.zeros(f1.shape[0]), np.ones(f2.shape[0]),
                       2 * np.ones(f2.shape[0])))
    kf = KFold(n_splits=10, shuffle=True)
    count_cm = 0
    for train, test in kf.split(x):
        x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
        cl = SVC(kernel='rbf', C=1)
        cl.fit(x_train, y_train)
        y_pred = cl.predict(x_test)
        if count_cm == 0: cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
        else: cm += (confusion_matrix(y_pred=y_pred, y_true=y_test))
        count_cm += 1
    print(cm)

