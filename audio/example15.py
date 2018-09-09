"""! 
@brief Example 15
@details librosa beattracking example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np
import scipy.io.wavfile as wavfile
import sys
import librosa

if __name__ == '__main__':
    # needs filepath as main argument:
    if len(sys.argv) != 2:
        sys.exit()
    # load file and extract tempo and beats:
    [Fs, s] = wavfile.read(sys.argv[1])
    tempo, beats = librosa.beat.beat_track(y=s, sr=Fs, units="time")
    beats -= 0.05
    # add small 220Hz sounds on the 2nd channel of the song on each beat
    s = s.reshape(-1, 1)
    s = np.array(np.concatenate((s, np.zeros(s.shape)), axis=1))
    for ib, b in enumerate(beats):
        t = np.arange(0, 0.2, 1.0 / Fs)
        amp_mod = 0.2 / (np.sqrt(t)+0.2) - 0.2
        amp_mod[amp_mod < 0] = 0
        x = s.max() * np.cos(2 * np.pi * t * 220) * amp_mod
        s[int(Fs * b):
          int(Fs * b) + int(x.shape[0]), 1] = x.astype('int16')
    wavfile.write("output.wav", Fs, np.int16(s))