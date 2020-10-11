"""! 
@brief Example 27
@details: Silence removal example (implemented in pyAudioAnalysis)
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import os, readchar
from pyAudioAnalysis.audioSegmentation import silence_removal as sR
from pyAudioAnalysis.audioBasicIO import read_audio_file

if __name__ == '__main__':
    # get non-silent segment limits:
    input_file = "../data/count.wav"
    fs, x = read_audio_file(input_file)
    seg_lims = sR(x, fs, 0.02, 0.01, 0.2, 0.5, True)
    print(seg_lims)
    # play each segment:
    for i_s, s in enumerate(seg_lims):
        print("Playing segment {0:d} of {1:d} "
              "({2:.2f} - {3:.2f} secs)".format(i_s, len(seg_lims), s[0], s[1]))
        # save the current segment to temp.wav
        os.system("ffmpeg -i {} -ss {} -t {} temp.wav "
                  "-loglevel panic -y".format(input_file, s[0], s[1]-s[0]))
        # play segment and wait for input
        os.system("play temp.wav")
        readchar.readchar()
