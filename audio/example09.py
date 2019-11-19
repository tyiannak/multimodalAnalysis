"""! 
@brief Example 09
@details pyAudioAnalysis feature extraction example for male / female speeches
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np
import plotly
import plotly.graph_objs as go
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO


if __name__ == '__main__':
    win = 0.05
    fp1 = "../data/general/speech/m1_neu-m1-l1.wav.wav" # male
    fp2 = "../data/general/speech/f1_neu-f1-l2.wav.wav" # female
    # read machine sound
    fs1, s1 = aIO.read_audio_file(fp1)
    fs2, s2 = aIO.read_audio_file(fp2)
    dur1, dur2 = len(s1) / float(fs1), len(s2) / float(fs2)
    # extract short term features
    [f1, fn] = aF.feature_extraction(s1, fs1, int(fs1 * win), int(fs1 * win))
    [f2, fn] = aF.feature_extraction(s2, fs2, int(fs2 * win), int(fs2 * win))
    figs = plotly.tools.make_subplots(rows=2, cols=2,
                                      subplot_titles=["male sig", "female sig",
                                                      fn[3], fn[3]])
    t1 = np.arange(0, dur1 - win, win)
    ts_1 = np.arange(0, dur1, 1/float(fs1))
    t2 = np.arange(0, dur2 - win, win)
    ts_2 = np.arange(0, dur2, 1/float(fs2))
    figs.append_trace(go.Scatter(x=ts_1, y=s1, showlegend=False), 1, 1)
    figs.append_trace(go.Scatter(x=ts_2, y=s2, showlegend=False), 1, 2)
    figs.append_trace(go.Scatter(x=t1, y=f1[3, :], showlegend=False), 2, 1)
    figs.append_trace(go.Scatter(x=t2, y=f2[3, :], showlegend=False), 2, 2)
    plotly.offline.plot(figs, filename="temp.html", auto_open=True)
