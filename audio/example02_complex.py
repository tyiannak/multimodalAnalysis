"""! 
@brief Example 02
@details Example of spectrogram computation for a wav file, using only scipy
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


def get_fft_spec(signal, fs, win):
    frame_size, signal_len, spec_real, spec_image, times = int(win * fs), len(signal), [], [], []
    # break signal into non-overlapping short-term windows (frames)
    frames = np.array([signal[x:x + frame_size] for x in
                       np.arange(0, signal_len - frame_size, frame_size)])
    for i_f, f in enumerate(frames): # for each frame
        times.append(i_f * win)
        # append mag of fft
        fft = scp.fft(f)
        print(np.real(fft))
        real = np.real(fft)
        imag = np.imag(fft)        
        freqs = np.arange(0, 1, 1.0/len(real)) * fs
        spec_real.append(real[0:int(len(real)/2)])
        spec_image.append(imag[0:int(len(real)/2)])        
        freqs = freqs[0:int(len(freqs)/2)]
    return np.array(spec_real).T, np.array(spec_image).T, freqs, times

if __name__ == '__main__':
    [Fs, s] = wavfile.read("../data/sample_music.wav")
    S_r, S_i, f, t = get_fft_spec(s, Fs, 0.02)
    heatmap = go.Heatmap(z=S_r, y=f, x=t)
    plotly.offline.plot(go.Figure(data=[heatmap], layout=layout),
                        filename="temp_r.html", auto_open=True)
    heatmap = go.Heatmap(z=S_i, y=f, x=t)
    plotly.offline.plot(go.Figure(data=[heatmap], layout=layout),
                        filename="temp_i.html", auto_open=True)                        
