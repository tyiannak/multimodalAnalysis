"""! 
@brief Example 15
@details beat-tracking toy example that demonstrates  a very simple onset
detection and tempo estimation method
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np
import scipy.io.wavfile as wavfile
import plotly
import plotly.graph_objs as go
from pyAudioAnalysis import ShortTermFeatures as aF

# read fil and extract features using pyAudioAnalysis
[Fs, s] = wavfile.read("../data/100_bpm.wav")
dur = s.shape[0] / Fs
win = 0.020
f, fn = aF.feature_extraction(s, Fs, win * Fs, win * Fs)
# keep energy and construct time arrays (for signal and feature seqs):
energy = f[fn.index("energy"), :]
t = np.arange(0, dur, 1.0 / Fs)
t_win = np.arange(win/2, dur, win)
# get local maxima as points that are larger than their neigbhors and > a thres
thres_energy = np.percentile(energy, 90)
peaks = np.nonzero((energy[1:-1] > energy[0:-2]) & (energy[1:-1] > energy[2:]) &
                   (energy[1:-1] > thres_energy))[0] + 1
# get period as most frequent value in successive peaks distances
period = int(np.median(peaks[1:] - peaks[0:-1]))
# compute tempo:
tempo_2 = 60./(period * win)
scores = []
# get points that satisfy the estimated tempo (beats):
for i in range(period):
    peaks_temp = np.arange(i, len(energy), period)
    scores.append(len(list(set(peaks).intersection(peaks_temp))))
peaks_f = np.arange(np.argmax(scores), len(energy), period)
# plot results
figs = plotly.tools.make_subplots(rows=2, cols=1, subplot_titles=["signal",
                                  "energy, onsets and beats"])
figs.append_trace(go.Scatter(x=t, y=s, showlegend=False), 1, 1)
figs.append_trace(go.Scatter(x=t_win, y=energy, name="energy"), 2, 1)
figs.append_trace(go.Scatter(x=t_win[peaks], y=energy[peaks],
                             mode='markers', name="onsets"), 2, 1)
figs.append_trace(go.Scatter(x=t_win[peaks_f], y=np.zeros(peaks_f.shape),
                             mode='markers', name="beats", marker = dict(
                             color = 'rgb(0, 100, 120)', size = 15,
                             line = dict(color = 'rgb(20, 120, 140)',
                                         width=5))), 2, 1)
plotly.offline.plot(figs, filename="temp.html", auto_open=True)