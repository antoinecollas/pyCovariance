import autograd.numpy as np
from pymanopt.manifolds import\
        ComplexEuclidean,\
        HermitianPositiveDefinite

from .base import Feature, make_feature_prototype

# ESTIMATION


def compute_scm(X, assume_centered=True):
    """ A function that computes the SCM for covariance matrix estimation
            Inputs:
                * X = a np.array of dim (p, N)
                with each observation along column dimension
                * assume_centered = bool.
                If False, data are centered with empirical mean.
            Outputs:
                * Sigma = the estimate"""
    _, N = X.shape
    if not assume_centered:
        mean = np.mean(X, axis=1, keepdims=True)
        X = X - mean
    scm = (X @ X.conj().T) / N
    return scm

# CLASSES


@make_feature_prototype
def covariance(assume_centered=True, **kwargs):
    p = kwargs['p']

    if assume_centered:
        name = 'Covariance'
    else:
        name = 'Centered_covariance'

    def _scm(X):
        return compute_scm(X, assume_centered=assume_centered)

    M = HermitianPositiveDefinite
    args_M = {'sizes': p}

    return Feature(name, _scm, M, args_M)


@make_feature_prototype
def covariance_euclidean(**kwargs):
    p = kwargs['p']

    name = 'Covariance_Euclidean'
    M = ComplexEuclidean
    args_M = {'sizes': p}
    return Feature(name, compute_scm, M, args_M)
