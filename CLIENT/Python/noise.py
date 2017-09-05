# Functions for generating noise 
from __future__ import division
import numpy as np

## Laplacian noise
def GenerateLaplaceNoise(length, eps, batchSize):
    scale = 4/eps/batchSize
    noise = np.random.laplace(loc = 0.0, scale = scale, size = length)
    return noise
