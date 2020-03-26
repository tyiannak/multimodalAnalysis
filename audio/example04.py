"""! 
@brief Example 04
@details pyAudioAnalysis spectrogram calculation and visualization example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np
import scipy.io.wavfile as wavfile
import plotly
import plotly.graph_objs as go
from pyAudioAnalysis import ShortTermFeatures as aF
layout = go.Layout(title='Spectrogram Extraction Example using pyAudioAnalysis',
                   xaxis=dict(title='time (sec)',),
                   yaxis=dict(title='Freqs (Hz)',))

def normalize_signal(signal):
    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    signal = (signal - signal.mean())
    return  signal / ((np.abs(signal)).max() + 0.0000000001)

if __name__ == '__main__':
    [Fs, s] = wavfile.read("../data/sample_music.wav")
    s = normalize_signal(s)
    [S, t, f] = aF.spectrogram(s, Fs, int(Fs * 0.020), int(Fs * 0.020))
    heatmap = go.Heatmap(z=S.T, y=f, x=t)
    plotly.offline.plot(go.Figure(data=[heatmap], layout=layout),
                        filename="temp.html", auto_open=True)