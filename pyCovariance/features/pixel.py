import numpy as np
from pymanopt.manifolds import Euclidean

from .base import Feature


# ESTIMATION


def get_center_vector(X):
    """ A function that ouputs center vector from a set of vectors.
            Inputs:
                * X = a np.array of dim (p, N)
                with each observation along column dimension
            Outputs:
                * center vector"""
    return X[:, X.shape[1]//2+1]


def compute_mean_vector(X):
    """ A function that ouputs mean of vectors.
            Inputs:
                * X = a np.array of dim (p, N)
                with each observation along column dimension
            Outputs:
                * mean vector"""
    return np.mean(X, axis=1)


def compute_intensity_vector(X):
    """ A function that intensity of center vector.
            Inputs:
                * X = a np.array of dim (p, N)
                with each observation along column dimension
            Outputs:
                * intensity"""
    x = get_center_vector(X)
    intensity = np.linalg.norm(x)
    return intensity


# CLASSES


def pixel_euclidean(p):
    name = 'Pixel_Euclidean'
    M = Euclidean
    args_M = [p]
    return Feature(name, get_center_vector, M, args_M)


def mean_pixel_euclidean(p):
    name = 'Mean_Pixel_Euclidean'
    M = Euclidean
    args_M = [p]
    return Feature(name, compute_mean_vector, M, args_M)


def intensity_euclidean():
    name = 'Intensity_Euclidean'
    M = Euclidean
    args_M = [1]
    return Feature(name, compute_intensity_vector, M, args_M)
