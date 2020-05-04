"""! 
@brief Example 20_roc
@details General sound classification example, with focus on ROC diagrams
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
from pyAudioAnalysis.audioTrainTest import extract_features_and_train
from pyAudioAnalysis.audioTrainTest import evaluate_model_for_folders
import os

if __name__ == '__main__':
    dirs = ["../data/general/train/animals",
            "../data/general/train/speech",
            "../data/general/train/objects",
            "../data/general/train/music"]
    class_names = [os.path.basename(d) for d in dirs]
    mw, stw = 2, .1
    extract_features_and_train(dirs, mw, mw, stw, stw, "svm_rbf",
                               "svm_general_4")

    dirs_test = ["../data/general/test/animals",
                 "../data/general/test/speech",
                 "../data/general/test/objects",
                 "../data/general/test/music"]

    evaluate_model_for_folders(dirs_test, "svm_general_4",
                               "svm_rbf", "animals")
