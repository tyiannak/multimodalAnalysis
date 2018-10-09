"""! 
@brief Example 19
@details Musical genre classification example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np, plotly, plotly.graph_objs as go
from pyAudioAnalysis.audioFeatureExtraction import dirWavFeatureExtraction as dW
from sklearn.svm import SVC
import utilities as ut
name_1, name_2 = "spectral_entropy_std", "chroma_std_std"
layout = go.Layout(title='Musical Genre Classification Example',
                   xaxis=dict(title=name_1,), yaxis=dict(title=name_2,))

if __name__ == '__main__':
    # get features from folders (all classes):
    f1, _, fn1 = dW("../data/musical_genres_8k/blues", 2, 1, 0.1, 0.1)
    f2, _, fn2 = dW("../data/musical_genres_8k/electronic", 2, 1, 0.1, 0.1)
    f3, _, fn3 = dW("../data/musical_genres_8k/jazz", 2, 1, 0.1, 0.1)
    # plot histograms for each feature and normalize
    ut.plot_feature_histograms([f1, f2, f3], fn1, ["blues", "electro", "jazz"])
    # concatenate features to extract overall mean and std ...
    f1 = np.array([f1[:, fn1.index(name_1)], f1[:, fn1.index(name_2)]]).T
    f2 = np.array([f2[:, fn1.index(name_1)], f2[:, fn1.index(name_2)]]).T
    f3 = np.array([f3[:, fn1.index(name_1)], f3[:, fn1.index(name_2)]]).T
    f = np.concatenate((f1, f2, f3), axis = 0)
    mean, std = f.mean(axis=0), np.std(f, axis=0)
    f1 = (f1 - mean) / std; f2 = (f2 - mean) / std;  f3 = (f3 - mean) / std
    f = (f - mean) / std
    # plot selected 2D features
    plt1 = go.Scatter(x=f1[:, 0], y=f1[:, 1], mode='markers', name="blues",
                      marker=dict(size=10,color='rgba(255, 182, 193, .9)',))
    plt2 = go.Scatter(x=f2[:, 0], y=f2[:, 1], mode='markers', name="electronic",
                      marker=dict(size=10,color='rgba(100, 182, 150, .9)',))
    plt3 = go.Scatter(x=f3[:, 0], y=f3[:, 1], mode='markers', name="jazz",
                      marker=dict(size=10,color='rgba(100, 100, 220, .9)',))
    # get classification decisions for grid
    y = np.concatenate((np.zeros(f1.shape[0]), np.ones(f2.shape[0]),
                       2 * np.ones(f3.shape[0])))
    cl = SVC(kernel='rbf', C=1)
    cl.fit(f, y)
    x_ = np.arange(f[:, 0].min(), f[:, 0].max(), 0.01)
    y_ = np.arange(f[:, 1].min(), f[:, 1].max(), 0.01)
    xx, yy = np.meshgrid(x_, y_)
    Z = cl.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape) / 2
    cs = go.Heatmap(x=x_, y=y_, z=Z, showscale=False,
                    colorscale= [[0, 'rgba(255, 182, 193, .3)'],
                                 [0.5, 'rgba(100, 182, 150, .3)'],
                                 [1, 'rgba(100, 100, 220, .3)']])
    plotly.offline.plot(go.Figure(data=[plt1, plt2, plt3, cs], layout=layout),
                        filename="temp2.html", auto_open=True)