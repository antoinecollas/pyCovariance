from pymanopt.manifolds import SymmetricPositiveDefinite

from .base import Feature

########## ESTIMATION ##########

def compute_scm(X):
    """ A function that computes the SCM for covariance matrix estimation
            Inputs:
                * X = a np.array of dim (N, p) with each observation along column dimension
            Outputs:
                * Sigma = the estimate"""
    (N, p) = X.shape
    return (X.conj().T @ X) / N

##########  CLASSES  ##########

def covariance(p):
    name = 'Covariance_Riemannian'
    M = SymmetricPositiveDefinite
    args_M = [p]
    return Feature(name, compute_scm, M, args_M)
