"""! 
@brief Example 06
@details librosa spectrogram calculation and visualization example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np
import scipy.io.wavfile as wavfile
import plotly
import librosa
import plotly.graph_objs as go
layout = go.Layout(title='Melgram Extraction Example using librosa',
                   xaxis=dict(title='time (sec)',),
                   yaxis=dict(title='Mel Coefficient (index)',))

def normalize_signal(signal):
    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    return (signal - signal.mean()) / ((np.abs(signal)).max() + 0.0000000001)

if __name__ == '__main__':
    [Fs, s] = wavfile.read("../data/sample_music.wav")
    s = normalize_signal(s)
    S = librosa.feature.melspectrogram(s, Fs, None, int(Fs * 0.020),
                                       int(Fs * 0.020), power=2)
    # create frequency and time axes
    f = range(S.shape[0])
    t = [float(t * int(Fs * 0.020)) / Fs for t in range(S.shape[1])]
    heatmap = go.Heatmap(z=S, y=f, x=t)
    plotly.offline.plot(go.Figure(data=[heatmap], layout=layout),
                        filename="temp.html", auto_open=True)