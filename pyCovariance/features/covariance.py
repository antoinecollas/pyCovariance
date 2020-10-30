import autograd.numpy as np
from multiprocessing import Process, Queue
from pymanopt.manifolds import HermitianPositiveDefinite
import warnings

from .base import BaseClassFeatures
from ..matrix_operators import *
from ..vectorization import *

########## ESTIMATION ##########

def SCM(X, *args):
    """ A function that computes the SCM for covariance matrix estimation
            Inputs:
                * X = a matrix of size p*N with each observation along column dimension
            Outputs:
                * Sigma = the estimate"""
    X = X.squeeze()
    (p, N) = X.shape
    return (X @ X.conj().T) / N

##########  CLASSES  ##########

class CovarianceEuclidean(BaseClassFeatures):
    def __init__(self):
        # code Hermitian manifold in Pymanopt
        super().__init__()

    def __str__(self):
        return 'Covariance_Euclidean'

    def estimation(self, X):
        return SCM(X)


class Covariance(BaseClassFeatures):
    def __init__(self, p):
        HPD = HermitianPositiveDefinite(p)
        super().__init__(manifold=HPD)

    def __str__(self):
        return 'Covariance_Riemannian'

    def estimation(self, X):
        return SCM(X)
