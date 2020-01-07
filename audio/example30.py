"""! 
@brief Example 30
@details: Music thumbnailing example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import os, readchar, matplotlib.pyplot as plt, matplotlib
from pyAudioAnalysis.audioBasicIO import read_audio_file, stereo_to_mono
from pyAudioAnalysis.audioSegmentation import musicThumbnailing

if __name__ == '__main__':
    # read signal and get normalized segment features:
    input_file = "../data/song2.mp3"
    fs, x = read_audio_file(input_file)
    x = stereo_to_mono(x)
    win = 0.5
    [A1, A2, B1, B2, Smatrix] = musicThumbnailing(x, fs, win, win, 20)
    os.system("avconv -i {} -ss {} -t {} thumb1.wav "
              "-loglevel panic -y".format(input_file, A1, A2 - A1))
    os.system("avconv -i {} -ss {} -t {} thumb2.wav "
              "-loglevel panic -y".format(input_file, B1, B2 - B1))
    print("Playing thumbnail 1"); os.system("play thumb1.wav -q")
    readchar.readchar()
    print("Playing thumbnail 2"); os.system("play thumb2.wav -q")
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="auto")
    plt.imshow(Smatrix)
    # Plot best-similarity diagonal:
    Xcenter = (A1 / win + A2 / win) / 2.0
    Ycenter = (B1 / win + B2 / win) / 2.0
    e1 = matplotlib.patches.Ellipse((Ycenter, Xcenter),
                                    20 * 1.4, 3, angle=45,
                                    linewidth=3, fill=False)
    ax.add_patch(e1)
    plt.plot([B1 / win, Smatrix.shape[0]], [A1 / win, A1 / win], color="k",
             linestyle="--", linewidth=2)
    plt.plot([B2 / win, Smatrix.shape[0]], [A2 / win, A2 / win], color="k",
             linestyle="--", linewidth=2)
    plt.plot([B1 / win, B1 / win], [A1 / win, Smatrix.shape[0]], color="k",
             linestyle="--", linewidth=2)
    plt.plot([B2 / win, B2 / win], [A2 / win, Smatrix.shape[0]], color="k",
             linestyle="--", linewidth=2)
    plt.xlim([0, Smatrix.shape[0]])
    plt.ylim([Smatrix.shape[1], 0])
    plt.show()