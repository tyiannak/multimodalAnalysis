"""! 
@brief Fingeprinting: show database stats
@details: Audio fingerprinting using dejavu
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

import MySQLdb as mysqldb

if __name__ == '__main__':
    db1 = mysqldb.connect("127.0.0.1", "root", "", 'dejavu')
    cursor = db1.cursor()
    a = cursor.execute("SELECT * FROM songs;")
    song_set = cursor.fetchall()
    for ir, r in enumerate(song_set):
        print("song {}\t {}".format(ir, r[1]))
    n_fings = cursor.execute("SELECT * FROM fingerprints;")
    n_figs_per_songs = n_fings / float(len(song_set))
    print(" - - - - - - - - - - - - - - - -")
    print("NUMBER OF SONGS IN DATABASE: {}".format(len(song_set)))
    print("NUMBER OF FINGERPRINTS IN DATABASE: {}".format(n_fings))
    print("NUMBER OF FINGERPRINTS PER SONG: {0:.1f}".format(n_figs_per_songs))
