"""! 
@brief fingerprinting search song
@details: Audio fingerprinting using dejavu
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

import sys
sys.path.append('../')
import dejavu
#from dejavu.recognize import FileRecognizer
from dejavu.logic.recognizer.file_recognizer import FileRecognizer

config = {
    "database": {
        "host": "127.0.0.1",
        "user": "root",
        "passwd": "",
        "db": "dejavu",
    }
}

if __name__ == '__main__':
    djv = dejavu.Dejavu(config)
    print(djv.recognize(FileRecognizer, "query.wav"))
