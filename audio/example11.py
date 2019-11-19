"""! 
@brief Example 11
@details pyAudioAnalysis chromagram example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import plotly
import plotly.graph_objs as go
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO
layout = go.Layout(title='Chromagram example for doremi.wav signal',
                   xaxis=dict(title='time (sec)',),
                   yaxis=dict(title='Chroma Name',))


if __name__ == '__main__':
    win = 0.04
    fp = "../data/doremi.wav" # music sample
    # read machine sound
    fs, s = aIO.read_audio_file(fp)
    fs = float(fs)
    dur1 = len(s) / float(fs)
    spec, time, freq = aF.chromagram(s, fs, int(fs * win),
                                       int(fs * win), False)
    heatmap = go.Heatmap(z=spec.T, y=freq, x=time)
    plotly.offline.plot(go.Figure(data=[heatmap], layout=layout),
                        filename="temp.html", auto_open=True)