"""! 
@brief Example 03
@details Example of audio recording and spectrogram computation
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import librosa
import example02
import scipy.fftpack as scp
import matplotlib.pyplot as plt
import numpy as np
import signal, sys, pyaudio, struct, cv2
import cv2
fs = 16000
bufSize = int(fs * 0.5)
def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
if __name__ == '__main__':
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16 , channels=1, rate=fs,
                     input=True, frames_per_buffer=bufSize)
    while 1:
        block = stream.read(bufSize, exception_on_overflow=False)
        s = np.array(list(struct.unpack("%dh"%(len(block)/2), block))) / (2**15)
        specgram, f, t = example02.get_fft_spec(s, fs, 0.02)
        iSpec = np.array(specgram[::-1] * 255, dtype=np.uint8)
        iSpec2 = cv2.resize(iSpec, (600, 300), interpolation=cv2.INTER_CUBIC)
        iSpec2 = cv2.applyColorMap(iSpec2, cv2.COLORMAP_JET)
        cv2.imshow('Spectrogram', iSpec2)
        cv2.moveWindow('Spectrogram', 50, 0 + 60)
        ch = cv2.waitKey(10)