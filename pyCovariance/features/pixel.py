import autograd.numpy as np

from .base import BaseClassFeatures
from ..vectorization import *

########## ESTIMATION ##########

##########  DISTANCE  ##########

##########   MEAN     ##########

def mean(X):
    """ Compute mean of vectors
        Inputs:
        --------
            * X = a (p, N) array where p is the dimension of data and N the number of samples used for estimation

        Outputs:
        ---------
            * 𝐱 = the feature for classification
        """
    mean = np.mean(X, axis=1)
    return mean


##########  CLASSES  ##########

class PixelEuclidean(BaseClassFeatures):
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return 'Pixel_Euclidean'
    
    def estimation(self, X):
        center_pixel = X[:, X.shape[1]//2+1, :]
        return center_pixel

    def distance(self, x1, x2):
        d = np.linalg.norm(x2-x1)
        d = np.real(d)
        return d

    def mean(self, X):
        return np.mean(X, axis=1)


class MeanPixelEuclidean(BaseClassFeatures):
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return 'Mean_Pixel_Euclidean'
    
    def estimation(self, X):
        return np.mean(X, 1)

    def distance(self, x1, x2):
        d = np.linalg.norm(x2-x1)
        d = np.real(d)
        return d

    def mean(self, X):
        return np.mean(X, axis=1)


class Intensity(BaseClassFeatures):
    def __init__(self):
        super().__init__()
    
    def __str__(self):
        return 'Intensity'
    
    def estimation(self, X):
        center_pixel = X[:, X.shape[1]//2+1, :]
        intensity = np.linalg.norm(center_pixel)
        return intensity

    def distance(self, x1, x2):
        d = np.linalg.norm(x2-x1)
        d = np.real(d)
        return d

    def mean(self, X):
        return np.mean(X, axis=1)
