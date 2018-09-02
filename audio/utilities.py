"""! 
@brief Utilities
@details General purpose routines
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import plotly
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_histograms(list_of_feature_mtr, feature_names, class_names):
    '''
    Plots the histograms of all classes and features for a given classification
    task.
    :param list_of_feature_mtr: list of feature matrices
                                (n_samples x n_features) each. for each class.
    :param feature_names:       list of feature names
    :param class_names:         list of class names (for each feature matrix)
    '''

    color_map = plt.cm.get_cmap('jet')
    n_features = len(feature_names)
    n_bins = 12
    n_columns = 4
    n_classes = len(class_names)
    n_rows = int(n_features / n_columns) + 1
    figs = plotly.tools.make_subplots(rows=n_rows, cols=n_columns,
                                      subplot_titles=feature_names)
    figs['layout'].update(height=(n_rows * 250))
    range_classes = range(int(int(255/n_classes)/2), 255, int(255/n_classes))
    clr = []
    for i in range(n_classes):
        clr.append('rgba({},{},{},{})'.format(color_map(range_classes[i])[0],
                                              color_map(range_classes[i])[1],
                                              color_map(range_classes[i])[2],
                                              color_map(range_classes[i])[3]))

    for i in range(n_features):
        # for each feature get its bin range (min:(max-min)/n_bins:max)
        f = np.vstack([x[:, i:i + 1] for x in list_of_feature_mtr])
        bins = np.arange(f.min(), f.max(), (f.max() - f.min()) / n_bins)
        for fi, f in enumerate(list_of_feature_mtr):
            # load the color for the current class (fi)
            mark_prop = dict(color=clr[fi], line=dict(color=clr[fi], width=3))
            # compute the histogram of the current feature (i) and normalize:
            h, _ = np.histogram(f[:, i], bins=bins)
            h = h.astype(float) / h.sum()
            cbins = (bins[0:-1] + bins[1:]) / 2
            scatter_1 = go.Scatter(x=cbins, y=h, name=class_names[fi],
                                   marker=mark_prop, showlegend=(i == 0))
            # (show the legend only on the first line)
            figs.append_trace(scatter_1, int(i/n_columns)+1, i % n_columns+1)

    for i in figs['layout']['annotations']:
        i['font'] = dict(size=10, color='#224488')
    plotly.offline.plot(figs, filename="report.html", auto_open=True)
