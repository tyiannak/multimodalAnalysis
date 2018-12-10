"""! 
@brief Fingerprinting training using dejavu
@details: Audio fingerprinting training using dejavu
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

import sys
sys.path.append('../')
import dejavu

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
    n_workers = 3
    djv.fingerprint_directory("songs", [".wav"], n_workers)
