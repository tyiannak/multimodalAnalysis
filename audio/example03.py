"""! 
@brief Example 03
@details Example of audio recording and spectrogram computation
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import example02
import numpy as np
import pyaudio, struct, cv2
fs = 8000
bufSize = int(fs * 1.0)
if __name__ == '__main__':
    # initialize recording:
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16 , channels=1, rate=fs,
                     input=True, frames_per_buffer=bufSize)
    while 1:
        # read recorded data, convert bytes to samples and then to numpy array
        block = stream.read(bufSize, exception_on_overflow=False)
        s = np.array(list(struct.unpack("%dh"%(len(block)/2), block))) / (2**15)
        # get spectrogram and visualize it using opencv
        specgram, f, t = example02.get_fft_spec(s, fs, 0.02)
        iSpec = np.array(specgram[::-1] * 255, dtype=np.uint8)
        iSpec2 = cv2.resize(iSpec, (600, 300), interpolation=cv2.INTER_CUBIC)
        iSpec2 = cv2.applyColorMap(iSpec2, cv2.COLORMAP_JET)
        cv2.imshow('Spectrogram', iSpec2)
        cv2.moveWindow('Spectrogram', 100, 100)
        ch = cv2.waitKey(10)