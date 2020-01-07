"""!
@brief Example 16
@details: Video diff estimation  example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import video_features as vf
import numpy as np
import plotly
import plotly.graph_objs as go

v = vf.VideoFeatureExtractor(["colors", "lbps", "hog"],
                             resize_width=300, step=0.5)
f, t, fn = v.extract_features("../data/dictator.mp4")

# normalize
m = f.mean(axis=0)
s = np.std(f, axis=0)
f2 = (f-m) / s
d = np.sum(np.abs(f2[1::] - f2[:-1]), axis = 1)
print(d.shape)
print(t[np.argmax(d)])

fig = go.Figure()
p1 = go.Scatter(x=t[1:], y=d, name='diffs')
fig = go.Figure(data = [p1])
plotly.offline.plot(fig, filename="temp.html", auto_open=True)