"""! 
@brief Example 26
@details: Regression example: song energy / danceability estimation
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np, csv, os, plotly, plotly.graph_objs as go
from pyAudioAnalysis.MidTermFeatures import directory_feature_extraction as dW
import utilities as ut
import joblib
import argparse

def parseArguments():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-t', '--target', nargs='+', required=True,
                        choices = ["energy", "dancability"])
    return parser.parse_args()

def get_hist_scatter(values, name):
    bins = np.arange(0, 1.1, 0.1)
    h_test = np.histogram(values, bins=bins)[0]
    h_test = h_test.astype(float) / h_test.sum()
    cbins = (bins[0:-1] + bins[1:]) / 2
    return go.Scatter(x=cbins, y=h_test, name=name)

if __name__ == '__main__':
    arg = parseArguments()
    target_type = arg.target[0]
    if os.path.isfile(target_type + ".bin"):
        x, y, filenames = joblib.load(target_type + ".bin")
    else:
        gt = {}
        with open('../data/music_data_small/{}.csv'.format(target_type)) \
                as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
               gt[row[0]] = row[1]
        f, f_names, fn1 = dW("../data/music_data_small", 2, 2, 0.1, 0.1)
        x, y, filenames = [], [], []
        for i_f, f_name in enumerate(f_names):
            if os.path.basename(f_name) in gt:
                x.append(f[i_f]); filenames.append(f_names)
                y.append(float(gt[os.path.basename(f_name)]))
        x = np.array(x)
        y = np.array(y)
        joblib.dump((x, y, filenames), target_type + ".bin")

    figs = plotly.tools.make_subplots(rows=1, cols=2,
                                      subplot_titles=["Distribution of real "
                                                      "y and predicted y",
                                                      "predicted (vert) "
                                                      "vs real (hor)"])
    mae, mae_r, all_pred, all_test = ut.svm_train_evaluate_regression(x, y,
                                                                      10, 1)
    sc1 = get_hist_scatter(all_pred, "pred")
    sc2 = get_hist_scatter(all_test, "real")
    figs.append_trace(sc1, 1, 1)
    figs.append_trace(sc2, 1, 1)
    plt2 = go.Scatter(x=all_test, y=all_pred, mode='markers', showlegend=False)
    figs.append_trace(plt2, 1, 2)
    plotly.offline.plot(figs, filename="temp.html", auto_open=True)
    print("MAE={0:.3f}\nMAE Baseline = {1:.3f}".format(mae, mae_r))
    print("Dataset STD (gt): {0:.2f}".format(all_test.std()))
