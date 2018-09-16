"""! 
@brief Example 16
@details librosa beattracking example: extract tempo and spectral centroid for
songs from different musical genres
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np
import scipy.io.wavfile as wavfile
import glob, os, librosa, plotly
from pyAudioAnalysis import audioFeatureExtraction as aF
import plotly.graph_objs as go
import utilities as ut


def get_dir_features(dir_name):
    feats = []
    for f in glob.glob(os.path.join(dir_name, "*.wav")):
        [Fs, s] = wavfile.read(f)
        tempo, _ = librosa.beat.beat_track(y=s, sr=Fs)
        f, _, fn = aF.mtFeatureExtraction(s, Fs, int(1.0 * Fs), int(1.0 * Fs),
                                          int(0.1 * Fs), int(0.1 * Fs))
        feats.append([tempo, np.mean(f[fn.index("spectral_centroid_mean")], axis=0)])
    return np.array(feats)


if __name__ == '__main__':
    g_paths = glob.glob("../data/musical_genres_small/*/")
    g_names = [p.split('/')[-2] for p in g_paths]
    clr = ut.get_color_combinations(len(g_paths))
    features = [get_dir_features(g) for g in g_paths]
    plots = [go.Scatter(x=features[i][:, 0], y=features[i][:, 1],
                        mode='markers', name=g_names[i], marker=dict(
                        color=clr[i], size=15))
             for i in range(len(g_paths))]
    plotly.offline.plot(plots, filename="temp.html", auto_open=True)