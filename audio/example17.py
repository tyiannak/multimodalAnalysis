"""! 
@brief Example 17
@details librosa pitch tracking example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import scipy.io.wavfile as wavfile
import librosa
import plotly
import numpy as np
import plotly.graph_objs as go
from scipy.signal import medfilt as mf

layout = go.Layout(title='Librosa pitch estimation',
                   xaxis=dict(title='time frame',),
                   yaxis=dict(title='freq (Hz)',))

def get_librosa_pitch(signal, fs, window):
    pitches, magnitudes = librosa.piptrack(y=signal.astype('float'), sr=fs, n_fft=int(window),
                                           hop_length=int(window/10))
    pitch_pos = np.argmax(magnitudes, axis=0)
    pitches_final = []
    for i in range(len(pitch_pos)):
        pitches_final.append(pitches[pitch_pos[i], i])
    pitches_final = np.array(pitches_final)
    pitches_final[pitches_final > 2000] = 0  # cut high pitches
    return mf(pitches_final, 3)              # use medfilt for smoothing

if __name__ == '__main__':
    [fs, s] = wavfile.read("../data/acapella.wav")
    p = get_librosa_pitch(s, fs, fs/20)
    plt = go.Scatter(x=np.arange(len(p)), y=p, mode='markers', showlegend=False)
    plotly.offline.plot(go.Figure(data=[plt], layout=layout),
                        filename="temp.html", auto_open=True)