"""! 
@brief Example 29
@details: Music segmentation example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import os, readchar, sklearn.cluster
from pyAudioAnalysis.MidTermFeatures import mid_feature_extraction as mT
from pyAudioAnalysis.audioBasicIO import read_audio_file, stereo_to_mono
from pyAudioAnalysis.audioSegmentation import flags2segs
from pyAudioAnalysis.audioTrainTest import normalizeFeatures

if __name__ == '__main__':
    # read signal and get normalized segment features:
    input_file = "../data/song1.mp3"
    fs, x = read_audio_file(input_file)
    x = stereo_to_mono(x)
    mt_size, mt_step, st_win = 5, 0.5, 0.05
    [mt_feats, st_feats, _] = mT(x, fs, mt_size * fs, mt_step * fs,
                                round(fs * st_win), round(fs * st_win * 0.5))
    (mt_feats_norm, MEAN, STD) = normalizeFeatures([mt_feats.T])
    mt_feats_norm = mt_feats_norm[0].T
    # perform clustering (k = 4)
    n_clusters = 4
    k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
    k_means.fit(mt_feats_norm.T)
    cls = k_means.labels_
    segs, c = flags2segs(cls, mt_step)      # convert flags to segment limits
    for sp in range(n_clusters):            # play each cluster's segment
        for i in range(len(c)):
            if c[i] == sp and segs[i, 1]-segs[i, 0] > 5:
                # play long segments of current cluster (only win_to_play seconds)
                d = segs[i, 1]-segs[i, 0]
                win_to_play = 10
                if win_to_play > d:
                    win_to_play = d
                print(" * * * * CLUSTER {0:d} * * * * * {1:.1f} - {2:.1f}, "
                      "playing {3:.1f}-{4:.1f}".format(c[i], segs[i, 0],
                                                      segs[i, 1],
                                                      segs[i, 0] + d/2 - win_to_play/2,
                                                      segs[i, 0] + d/2 + win_to_play/2))
                cmd = "avconv -i {} -ss {} -t {} temp.wav " \
                          "-loglevel panic -y".format(input_file,
                                                      segs[i, 0] + d/2 - win_to_play/2,
                                                      win_to_play)
                os.system(cmd)
                os.system("play temp.wav -q")
                readchar.readchar()