"""! 
@brief Example 08
@details pyAudioAnalysis feature extraction example 1
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np
import plotly
import plotly.graph_objs as go
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO


if __name__ == '__main__':
    # read machine sound
    fs, s = aIO.read_audio_file("../data/activity_sounds/1-46744-A.ogg.wav")
    duration = len(s) / float(fs)
    # extract short term features and plot ZCR and Energy
    [f, fn] = aF.feature_extraction(s, fs, int(fs * 0.050), int(fs * 0.050))
    figs = plotly.tools.make_subplots(rows=3, cols=1,
                                      subplot_titles=["signal", fn[0], fn[1]])
    time = np.arange(0, duration - 0.050, 0.050)
    time_s = np.arange(0, duration, 1/float(fs))
    figs.append_trace(go.Scatter(x=time_s, y=s, showlegend=False), 1, 1)
    figs.append_trace(go.Scatter(x=time, y=f[0, :], showlegend=False), 2, 1)
    figs.append_trace(go.Scatter(x=time, y=f[1, :], showlegend=False), 3, 1)
    plotly.offline.plot(figs, filename="temp.html", auto_open=True)
