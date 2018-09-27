"""! 
@brief Example 18
@details pitch ability to discriminate between male and female speech
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import scipy.io.wavfile as wavfile
import os
import librosa
import plotly
import numpy as np
import glob
import plotly.graph_objs as go
import example17

def get_dir_features(dir_name):
    feats = []
    for f in glob.glob(os.path.join(dir_name, "*.wav")):
        [Fs, s] = wavfile.read(f)
        p = example17.get_librosa_pitch(s, Fs, Fs / 20)
        feats.append(np.mean(p[p>0]))
    return feats

if __name__ == '__main__':
    feats1 = get_dir_features("../data/gender/male/")
    feats2 = get_dir_features("../data/gender/female/")

    print(np.mean(feats1))
    print(np.mean(feats2))
