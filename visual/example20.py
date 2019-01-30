"""!
@brief Example 20
@details: Pytorch simple example
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
from __future__ import print_function
import torch
import numpy as np

print(torch.rand(2, 3))
print(torch.empty(2, 3))
print(torch.ones(2, 3))
print(torch.zeros(2, 3))
print(torch.zeros(2, 3).numpy())
print(torch.from_numpy(np.ones(3)))
