from pymanopt.manifolds import HermitianPositiveDefinite

from .base import Feature, feature_estimation

########## ESTIMATION ##########

@feature_estimation
def SCM(X):
    """ A function that computes the SCM for covariance matrix estimation
            Inputs:
                * X = a np.array of dim (p, N) with each observation along column dimension
            Outputs:
                * Sigma = the estimate"""
    (p, N) = X.shape
    return (X @ X.conj().T) / N

##########  CLASSES  ##########

def Covariance(p):
    name = 'Covariance_Riemannian'
    M = HermitianPositiveDefinite
    args_M = [p]
    return Feature(name, SCM, M, args_M)
