import autograd.numpy as np
from pymanopt.manifolds import\
        ComplexEuclidean,\
        HermitianPositiveDefinite

from .base import Feature, make_feature_prototype
from ..manifolds import ComplexGaussianIG

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
def covariance(assume_centered=True, min_grad_norm_mean=1e-8, p=None, N=None):
    if assume_centered:
        name = 'Covariance'
    else:
        name = 'Centered_covariance'

    def _scm(X):
        return compute_scm(X, assume_centered=assume_centered)

    M = HermitianPositiveDefinite
    args_M = {'sizes': p}

    return Feature(name, _scm, M, args_M,
                   min_grad_norm_mean=min_grad_norm_mean)


@make_feature_prototype
def covariance_euclidean(assume_centered=True, p=None, N=None):
    if assume_centered:
        name = 'Covariance_Euclidean'
    else:
        name = 'Centered_covariance_Euclidean'

    def _scm(X):
        return compute_scm(X, assume_centered=assume_centered)

    M = ComplexEuclidean
    args_M = {'sizes': (p, p)}

    return Feature(name, _scm, M, args_M)


@make_feature_prototype
def covariance_div_alpha(
    assume_centered=True,
    div_alpha_real_case=False,
    symmetrize_div=False,
    alpha=0.5,
    min_grad_norm_mean=1e-7,
    p=None,
    N=None
):
    if assume_centered:
        name = 'Covariance_div_alpha_'
    else:
        name = 'Centered_covariance_div_alpha_'
    if symmetrize_div:
        name += 'sym_'
    name += str(alpha)

    def _scm(X):
        return compute_scm(X, assume_centered=assume_centered)

    # define a manifold with the sym alpha divergence:
    # it leverages the sym alpha divergence of the
    # ComplexGaussianIG manifold and the HermitianPositiveDefinite
    # manifold to compute barycentres
    class M(HermitianPositiveDefinite):
        def __init__(self, p, k=1, alpha=0.5):
            self._alpha = alpha
            self._M_div_alpha = ComplexGaussianIG(p=p, k=k, alpha=alpha)
            super().__init__(p, k=k)

        # remove exp and log: we only need a retraction to compute barycentres
        def exp(self, x, u):
            raise NotImplementedError

        def log(self, x, y):
            raise NotImplementedError

        def dist(self, x1, x2):
            # BE CAREFUL: this is not a distance
            # it is a symmmetric divergence
            x1 = [np.zeros(shape=(*(x1.shape[:-1]), 1), dtype=x1.dtype), x1]
            x2 = [np.zeros(shape=(*(x2.shape[:-1]), 1), dtype=x2.dtype), x2]
            if symmetrize_div:
                if div_alpha_real_case:
                    return self._M_div_alpha.div_alpha_sym_real_case(x1, x2)
                else:
                    return self._M_div_alpha.div_alpha_sym(x1, x2)
            else:
                if div_alpha_real_case:
                    return self._M_div_alpha.div_alpha_real_case(x1, x2)
                else:
                    return self._M_div_alpha.div_alpha(x1, x2)

    args_M = {'sizes': p, 'alpha': alpha}

    return Feature(name, _scm, M, args_M,
                   min_grad_norm_mean=min_grad_norm_mean)
