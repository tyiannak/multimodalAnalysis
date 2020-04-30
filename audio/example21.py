"""! 
@brief Example 21
@details Overfitting Example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np, plotly, plotly.graph_objs as go
from sklearn.svm import SVC

def demo(f1, f2, c):
    f = np.concatenate((f1, f2), axis = 0)
    plt1 = go.Scatter(x=f1[:, 0], y=f1[:, 1], mode='markers', name="c1",
                      marker=dict(size=10, color='rgba(200, 0, 100, .9)', ),
                      showlegend=False)
    plt2 = go.Scatter(x=f2[:, 0], y=f2[:, 1], mode='markers', name="c2",
                      marker=dict(size=10, color='rgba(0, 200, 100, .9)', ),
                      showlegend=False)
    y = np.concatenate((np.zeros(f1.shape[0]), np.ones(f2.shape[0])))
    cl = SVC(kernel='rbf', C=c, gamma='auto')
    cl.fit(f, y)
    x_ = np.arange(f[:, 0].min()-0.01, f[:, 0].max()+0.01, 0.005)
    y_ = np.arange(f[:, 1].min()-0.01, f[:, 1].max()+0.01, 0.005)
    xx, yy = np.meshgrid(x_, y_)
    Z = cl.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    cs = go.Heatmap(x=x_, y=y_, z=Z, showscale=False,
                    colorscale= [[0, 'rgba(200, 0, 100, .3)'],
                                 [1, 'rgba(0, 200, 100, .3)']])
    return plt1, plt2, cs

if __name__ == '__main__':
    # get features from folders (all classes):
    f11 = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
                    [0, 1], [1, 1], [2, 1], [3, 1], [4, 1],
                    [0, 2], [1, 2], [2, 2], [3, 2], [4, 2],
                    [0, 3], [1, 3], [2, 3], [3, 3], [4, 3],
                    [0, 4], [1, 4], [2, 4], [3, 4], [4, 4], ]) / 10.0
    f21 = np.array([[5, 0], [6, 0], [7, 0],
                    [5, 1], [6, 1], [7, 1],
                    [5, 2], [6, 2], [7, 2],
                    [5, 3], [6, 3], [7, 3],
                    [5, 4], [6, 4], [7, 4],
                    [0, 5], [1, 5], [2, 5], [3, 5], [4, 5],
                    [5, 5], [6, 5], [7, 5],
                    [0, 6], [1, 6], [2, 6], [3, 6], [4, 6],
                    [5, 6], [6, 6], [7, 6],
                    [0, 7], [1, 7], [2, 7], [3, 7], [4, 7],
                    [5, 7], [6, 7], [7, 7],
                    ]) / 10.0
    f12 = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
                    [0, 1], [1, 1], [2, 1], [3, 1], [4, 1],
                    [0, 2], [1, 2], [2, 2], [3, 2], [4, 2],
                    [0, 3], [1, 3], [2, 3], [3, 3], [4, 3],
                    [0, 4], [1, 4], [2, 4], [3, 4], [4, 4], ]) / 10.0
    f22 = np.array([[5, 0], [6, 0], [7, 0],
                    [5, 1], [6, 1], [7, 1],
                    [5, 2], [6, 2], [7, 2],
                    [5, 3], [6, 3], [7, 3],
                    [5, 4], [6, 4], [7, 4],
                    [0, 5], [1, 5], [2, 5], [3, 5], [4, 5],
                    [5, 5], [6, 5], [7, 5],
                    [0, 6], [1, 6], [2, 6], [3, 6], [4, 6],
                    [5, 6], [6, 6], [7, 6],
                    [0, 7], [1, 7], [2, 7], [3, 7], [4, 7],
                    [5, 7], [6, 7], [7, 7], [1.5, 3]
                    ]) / 10.0
    Cs = [10, 1000, 1000000000]
    n_r, n_c = len(Cs), 2
    titles = []
    for i in range(len(Cs)):
        titles.append('dataset 1 , C = {}'.format(Cs[i]))
        titles.append('dataset 2 , C = {}'.format(Cs[i]))

    figs = plotly.subplots.make_subplots(rows=n_r, cols=n_c,
                                         subplot_titles=titles)
    for ic, c in enumerate(Cs):
        p1, p2, cs = demo(f11, f21, c)
        figs.append_trace(p1, ic + 1, 1)
        figs.append_trace(p2, ic + 1, 1)
        figs.append_trace(cs, ic + 1, 1)
        p1, p2, cs = demo(f12, f22, c)
        figs.append_trace(p1, ic + 1, 2)
        figs.append_trace(p2, ic + 1, 2)
        figs.append_trace(cs, ic + 1, 2)
    plotly.offline.plot(figs, filename="temp.html", auto_open=True)

