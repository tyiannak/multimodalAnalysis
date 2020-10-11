"""! 
@brief Fingerprinting online
@details
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import numpy as np
import scipy.io.wavfile as wavfile
import pyaudio, struct, sys
sys.path.append('../')
import dejavu
from dejavu.logic.recognizer.file_recognizer import FileRecognizer

config = {
    "database": {
        "host": "127.0.0.1",
        "user": "root",
        "passwd": "",
        "db": "dejavu",
    }
}

fs = 8000
bufSize = int(fs * 0.1)
dur = 2

if __name__ == '__main__':
    pa = pyaudio.PyAudio() # initialize recording
    stream = pa.open(format=pyaudio.paInt16 , channels=1, rate=fs,
                     input=True, frames_per_buffer=bufSize)
    djv = dejavu.Dejavu(config)
    aggregated_buf = np.array([])
    while 1:
        # read recorded data, convert bytes to samples and then to numpy array
        block = stream.read(bufSize, exception_on_overflow=False)
        s = np.array(list(struct.unpack("%dh"%(len(block)/2), block))).astype(float)
        aggregated_buf = np.concatenate((aggregated_buf, s))
        if aggregated_buf.shape[0] > fs * dur:
            aggregated_buf = (2**15) * aggregated_buf / aggregated_buf.max()
            wavfile.write("temp.wav", fs, np.int16(aggregated_buf))
            aggregated_buf = np.array([])
            response = djv.recognize(FileRecognizer, "temp.wav")

            if len(response["results"]) > 0:
                if response["results"][0]["fingerprinted_confidence"] > 0.5:
                    print(response["results"][0]["song_name"],
                          response["results"][0]["fingerprinted_confidence"])
                else:
                    print(response["results"][0])
