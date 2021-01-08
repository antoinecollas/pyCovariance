from pymanopt.manifolds import\
        ComplexEuclidean,\
        HermitianPositiveDefinite

from .base import Feature

# ESTIMATION


def compute_scm(X):
    """ A function that computes the SCM for covariance matrix estimation
            Inputs:
                * X = a np.array of dim (p, N)
                with each observation along column dimension
            Outputs:
                * Sigma = the estimate"""
    (p, N) = X.shape
    scm = (X @ X.conj().T) / N
    return scm

# CLASSES


def covariance(p):
    name = 'Covariance'
    M = HermitianPositiveDefinite
    args_M = {'sizes': p}
    return Feature(name, compute_scm, M, args_M)


def covariance_euclidean(p):
    name = 'Covariance_Euclidean'
    M = ComplexEuclidean
    args_M = {'sizes': p}
    return Feature(name, compute_scm, M, args_M)
