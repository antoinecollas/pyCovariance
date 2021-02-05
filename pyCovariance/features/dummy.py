import numpy as np
import numpy.linalg as la
from pymanopt.manifolds import ComplexEuclidean, Euclidean

from .base import Feature, make_feature_prototype


# ESTIMATION


def identity(X):
    """ Identity function.
            Inputs:
                * X = a np.array of dim (p, N)
                with each observation along column dimension
            Outputs:
                * X"""
    return X


def get_center_vector(X):
    """ A function that ouputs center vector from a set of vectors.
            Inputs:
                * X = a np.array of dim (p, N)
                with each observation along column dimension
            Outputs:
                * center vector"""
    return X[:, X.shape[1]//2]


def compute_mean_vector(X):
    """ A function that ouputs mean of vectors.
            Inputs:
                * X = a np.array of dim (p, N)
                with each observation along column dimension
            Outputs:
                * mean vector"""
    return np.mean(X, axis=1)


def compute_intensity_center_vector(X):
    """ A function that computes intensity of center vector.
            Inputs:
                * X = a np.array of dim (p, N)
                with each observation along column dimension
            Outputs:
                * intensity"""
    x = get_center_vector(X)
    intensity = la.norm(x)
    return intensity


def compute_intensity_vector(X):
    """ A function that computes a size N intensity vector.
            Inputs:
                * X = a np.array of dim (p, N)
                with each observation along column dimension
            Outputs:
                * intensity vector"""
    intensity = la.norm(X, axis=0)
    return intensity


# CLASSES


@make_feature_prototype
def identity_euclidean(**kwargs):
    p = kwargs['p']
    N = kwargs['N']

    name = 'Identity_Euclidean'
    M = ComplexEuclidean
    args_M = {'sizes': (p, N)}
    return Feature(name, identity, M, args_M)


@make_feature_prototype
def center_euclidean(**kwargs):
    p = kwargs['p']

    name = 'Center_Euclidean'
    M = ComplexEuclidean
    args_M = {'sizes': p}
    return Feature(name, get_center_vector, M, args_M)


@make_feature_prototype
def center_intensity_euclidean(**kwargs):
    name = 'Center_Intensity_Euclidean'
    M = Euclidean
    args_M = {'sizes': 1}
    return Feature(name, compute_intensity_center_vector, M, args_M)


@make_feature_prototype
def mean_vector_euclidean(**kwargs):
    p = kwargs['p']

    name = 'Mean_Vector_Euclidean'
    M = ComplexEuclidean
    args_M = {'sizes': p}
    return Feature(name, compute_mean_vector, M, args_M)


@make_feature_prototype
def intensity_vector_euclidean(**kwargs):
    N = kwargs['N']

    name = 'Intensity_Euclidean'
    M = Euclidean
    args_M = {'sizes': N}
    return Feature(name, compute_intensity_vector, M, args_M)
