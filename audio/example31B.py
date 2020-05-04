"""! 
@brief Example 31B
@details: Speech music discrimination and segmentation (using a trained
speech - music segment classifier)
Important: Need to run 31A first to extract speech music model (stored
in svm_speech_music)
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
from pyAudioAnalysis.audioSegmentation import mid_term_file_classification

if __name__ == '__main__':
    au = "../data/scottish_radio.wav"
    gt = "../data/scottish_radio.segments"
#    au = "../data/musical_genres_small/hiphop/run_dmc_peter_riper.wav"
    mid_term_file_classification(au, "svm_speech_music", "svm_rbf", True, gt)
