"""! 
@brief Example 02
@details Example of spectrogram computation for a wav file, using only fft
from scipy.
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import scipy.fftpack as scp
import numpy as np
import scipy.io.wavfile as wavfile
import plotly
import plotly.graph_objs as go
layout = go.Layout(title='Spectrogram Calculation Example',
                   xaxis=dict(title='time (sec)',),
                   yaxis=dict(title='Freqs (Hz)',))
if __name__ == '__main__':
    [fs, s] = wavfile.read("../data/sample_music.wav")
    win = 0.02
    frame_size, signal_len, spec, time_axis = int(win * fs), len(s), [], []
    # break signal into non-overlapping short-term windows (frames)
    frames = np.array([s[x:x + frame_size] for x in
                       np.arange(0, signal_len - frame_size, frame_size)])
    for i_f, f in enumerate(frames): # for each frame
        time_axis.append(i_f * win)
        # append mag of fft
        X = np.abs(scp.fft(f)) ** 2
        freqs = np.arange(0, 1, 1.0/len(X)) * (fs/2)
        spec.append(X[0:int(len(X)/2)] / X.max())
    spec = np.array(spec).T
    heatmap = go.Heatmap(z=spec, y=freqs, x=time_axis)
    plotly.offline.plot(go.Figure(data=[heatmap], layout=layout),
                        filename="temp.html", auto_open=True)