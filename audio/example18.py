"""! 
@brief Example 18
@details speech music classification example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np
import plotly
import plotly.graph_objs as go
from pyAudioAnalysis import audioFeatureExtraction as aF
from sklearn.svm import SVC
name_1, name_2 = "energy_entropy_mean", "zcr_std"
layout = go.Layout(title='Speech Music Classification Example',
                   xaxis=dict(title=name_1,),yaxis=dict(title=name_2,))

if __name__ == '__main__':
    f1, _, fn1 = aF.dirWavFeatureExtraction("../data/speech_music/speech", 1, 1, 0.1, 0.1)
    f1 = np.array([f1[:, fn1.index(name_1)], f1[:, fn1.index(name_2)]]).T
    f2, _, fn2 = aF.dirWavFeatureExtraction("../data/speech_music/music", 1, 1, 0.1, 0.1)
    f2 = np.array([f2[:, fn1.index(name_1)], f2[:, fn1.index(name_2)]]).T
    f = np.concatenate((f1, f2), axis = 0)
    mean, std = f.mean(axis = 0), np.std(f, axis=0)
    f1 = (f1 - mean) / std; f2 = (f2 - mean) / std; f = (f - mean) / std
    plt1 = go.Scatter(x=f1[:, 0], y=f1[:, 1], mode='markers', name="speech")
    plt2 = go.Scatter(x=f2[:, 0], y=f2[:, 1], mode='markers', name="music")
    y = np.concatenate((np.zeros(f1.shape[0]), np.ones(f2.shape[0])))
    cl = SVC(kernel='rbf', C=10)
    cl.fit(f, y)
    x_ = np.arange(f[:,0].min(), f[:,0].max(), 0.01)
    y_ = np.arange(f[:,1].min(), f[:,1].max(), 0.01)
    xx, yy = np.meshgrid(x_, y_)
    Z = cl.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = go.Heatmap(x=x_, y=y_, z=Z, showscale=False)
    plotly.offline.plot(go.Figure(data=[plt1, plt2, cs], layout=layout),
                        filename="temp.html", auto_open=True)