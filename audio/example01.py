"""! 
@brief Example 01 
@details 
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
@copyright NCSR Demokritos
"""

import scipy.fftpack as scp
import numpy as np
import plotly
import plotly.graph_objs as go


def main():
    f1 = 500
    f2 = 2500
    fs = 8000
    duration = 0.1
    # define time range
    t = np.arange(0, duration, 1.0/fs)
    # define signal as sum of cosines
    x = np.cos(2 * np.pi * t * f1) + np.cos(2 * np.pi * t * f2)
    # get mag of fft
    X = np.abs(scp.fft(x))
    # normalize FFT mag
    X = X / X.max()
    freqs = np.arange(0, 1, 1.0/len(X)) * (fs)
    # get 1st symmetric part
    freqs_1 = freqs[0:int(len(freqs)/2)]
    X_1 = X[0:int(len(X)/2)]
    figs = plotly.tools.make_subplots(rows=2, cols=1,
                                      subplot_titles=["FFT Mag",
                                                      "FFT Mag 1st sym part"])
    figs.append_trace(go.Scatter(x=freqs, y=X, showlegend=False), 1, 1)
    figs.append_trace(go.Scatter(x=freqs_1, y=X_1, showlegend=False), 2, 1)
    plotly.offline.plot(figs, filename="temp.html", auto_open=True)

if __name__ == '__main__':
    main()