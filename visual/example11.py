"""!
@brief Example 11
@details: Visual features visualization for 2 images using plotly.
(Color features)
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""

from visual_features import ImageFeatureExtractor
import plotly
import plotly.graph_objs as go
# given two filenames
filename1 = '../data/images_general/beach.jpg'
filename2 = '../data/images_general/new_york.jpg'
# initialize an image feature extractor for color features:
ife = ImageFeatureExtractor(list_of_features=["colors"])
# extract features for both images:
f1, f1n = ife.extract_features(file_path=filename1)
f2, f2n = ife.extract_features(file_path=filename2)
# plot both feature vectors in plotly:
fig = go.Figure()
p1 = go.Scatter(x=f1n, y=f1, name=filename1)
p2 = go.Scatter(x=f2n, y=f2, name=filename2)
fig = go.Figure(data = [p1, p2])
plotly.offline.plot(fig, filename="temp.html", auto_open=True)

