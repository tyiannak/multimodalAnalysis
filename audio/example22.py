"""! 
@brief Example 22
@details Audio event detection: features discrimination and
2D-feature classification
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np, plotly, plotly.graph_objs as go
from pyAudioAnalysis.MidTermFeatures import directory_feature_extraction as dW
from sklearn.svm import SVC
import utilities as ut
name_1, name_2 = "spectral_spread_std", "mfcc_5_mean"
layout = go.Layout(title='Activity Detection Example',
                   xaxis=dict(title=name_1,), yaxis=dict(title=name_2,))

if __name__ == '__main__':
    # get features from folders (all classes):
    f1, _, fn1 = dW("../data/activity_sounds/cupboards", 1, 1, 0.05, 0.05)
    f2, _, fn1 = dW("../data/activity_sounds/door", 1, 1, 0.05, 0.05)
    f3, _, fn1 = dW("../data/activity_sounds/silence", 1, 1, 0.05, 0.05)
    f4, _, fn1 = dW("../data/activity_sounds/walk", 1, 1, 0.05, 0.05)

    # plot histograms for each feature and normalize
    ut.plot_feature_histograms([f1, f2, f3, f4], fn1, ["cupboards", "door",
                                                   "silence", "walk"])
    # concatenate features to extract overall mean and std ...
    f1 = np.array([f1[:, fn1.index(name_1)], f1[:, fn1.index(name_2)]]).T
    f2 = np.array([f2[:, fn1.index(name_1)], f2[:, fn1.index(name_2)]]).T
    f3 = np.array([f3[:, fn1.index(name_1)], f3[:, fn1.index(name_2)]]).T
    f4 = np.array([f4[:, fn1.index(name_1)], f4[:, fn1.index(name_2)]]).T

    f = np.concatenate((f1, f2, f3, f4), axis = 0)
    mean, std = f.mean(axis=0), np.std(f, axis=0)
    f1 = (f1 - mean) / std; f2 = (f2 - mean) / std
    f3 = (f3 - mean) / std; f4 = (f4 - mean) / std
    f = (f - mean) / std
    # plot selected 2D features
    plt1 = go.Scatter(x=f1[:, 0], y=f1[:, 1], mode='markers', name="cupboards",
                      marker=dict(size=10,color='rgba(180, 0, 0, .9)',))
    plt2 = go.Scatter(x=f2[:, 0], y=f2[:, 1], mode='markers', name="door",
                      marker=dict(size=10,color='rgba(0, 180, 0, .9)',))
    plt3 = go.Scatter(x=f3[:, 0], y=f3[:, 1], mode='markers', name="silence",
                      marker=dict(size=10,color='rgba(0, 0, 180, .9)',))
    plt4 = go.Scatter(x=f4[:, 0], y=f4[:, 1], mode='markers', name="walk",
                      marker=dict(size=10,color='rgba(255, 255, 155, .9)',))
    # get classification decisions for grid
    y = np.concatenate((np.zeros(f1.shape[0]), np.ones(f2.shape[0]),
                       2 * np.ones(f3.shape[0]), 3 * np.ones(f4.shape[0])))
    cl = SVC(kernel='rbf', C=1)
    cl.fit(f, y)
    x_ = np.arange(f[:, 0].min(), f[:, 0].max(), 0.01)
    y_ = np.arange(f[:, 1].min(), f[:, 1].max(), 0.01)
    xx, yy = np.meshgrid(x_, y_)
    Z = cl.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape) / 3
    cs = go.Heatmap(x=x_, y=y_, z=Z, showscale=False,
                    colorscale= [[0, 'rgba(180, 0, 0, .3)'],
                                 [0.33, 'rgba(0, 180, 0, .3)'],
                                 [0.66, 'rgba(0, 0, 180, .3)'],
                                 [1, 'rgba(150, 150, 0, .3)']])
    plotly.offline.plot(go.Figure(data=[plt1, plt2, plt3, plt4, cs], layout=layout),
                        filename="temp2.html", auto_open=True)