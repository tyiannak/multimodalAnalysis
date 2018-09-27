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

def get_librosa_pitch(signal, fs, window):
    pitches, magnitudes = librosa.piptrack(y=signal, sr=fs, n_fft=int(window),
                                           hop_length=int(window/10))
    pitch_pos = np.argmax(magnitudes, axis=0)
    pitches_final = []
    for i in range(len(pitch_pos)):
        pitches_final.append(pitches[pitch_pos[i], i])
    pitches_final = np.array(pitches_final)
    pitches_final[pitches_final > 500] = 0
    return mf(pitches_final, 3)

if __name__ == '__main__':
    [fs, s1] = wavfile.read("../data/speech_male_sample.wav")
    [fs, s2] = wavfile.read("../data/speech_female_sample.wav")
    p1 = get_librosa_pitch(s1, fs, fs/20)
    p2 = get_librosa_pitch(s2, fs, fs/20)
    plt1 = go.Scatter(x=np.arange(len(p1)), y=p1, mode='markers',
                      showlegend=False)
    plt2 = go.Scatter(x=np.arange(len(p2)), y=p2, mode='markers',
                      showlegend=False)
    figs = plotly.tools.make_subplots(rows=1, cols=2,
                                      subplot_titles=["male pitch",
                                                      "female pitch"])
    figs.append_trace(plt1, 1, 1)
    figs.append_trace(plt2, 1, 2)
    plotly.offline.plot(figs, filename="temp.html", auto_open=True)