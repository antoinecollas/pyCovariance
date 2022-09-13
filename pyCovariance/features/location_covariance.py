import autograd.numpy as np

from .base import Feature, make_feature_prototype
from ..manifolds import ComplexGaussianIG

# Gaussian estimation


def Gaussian_estimation(X):
    """ Function that computes Gaussian estimators: sample mean and SCM
        Inputs:
            * X = a matrix of size p*N
        Outputs:
            * mu = estimate of location
            * sigma = estimate of covariance"""
    p, N = X.shape
    mu = np.mean(X, axis=1).reshape((-1, 1))
    sigma = (1/N) * (X-mu) @ (X-mu).conj().T
    return mu, sigma

# CLASSES


@make_feature_prototype
def location_covariance_orth_triangle(
    min_grad_norm_mean=1e-6,
    p=None,
    N=None
):
    name = 'Location_covariance_orth_triangle'

    class M(ComplexGaussianIG):
        def dist(self, x1, x2):
            return self.div_orth(x1, x2)

    args_M = {'sizes': p}

    # init mean
    # argcosh(1 + x) is not differentiable at x = 0.
    # Hence we have to provide an initialization that is not a (mu, sigma)
    # among the parameters to cluster/classify.
    def init_mean(X):
        theta = [
            np.mean(X[0], axis=0),
            np.mean(X[1], axis=0)
        ]
        return theta

    return Feature(name, Gaussian_estimation, M, args_M,
                   init_mean, min_grad_norm_mean=min_grad_norm_mean)


@make_feature_prototype
def location_covariance_scale_triangle(
    min_grad_norm_mean=1e-6,
    p=None,
    N=None
):
    name = 'Location_covariance_scale_triangle'

    class M(ComplexGaussianIG):
        def dist(self, x1, x2):
            return self.div_scale(x1, x2)

    args_M = {'sizes': p}

    def init_mean(X):
        theta = [
            np.mean(X[0], axis=0),
            np.mean(X[1], axis=0)
        ]
        return theta

    return Feature(name, Gaussian_estimation, M, args_M,
                   init_mean, min_grad_norm_mean=min_grad_norm_mean)


@make_feature_prototype
def location_covariance_div_alpha(
    alpha=0.5,
    div_alpha_real_case=False,
    symmetrize_div=False,
    min_grad_norm_mean=1e-6,
    p=None,
    N=None
):
    name = 'Location_covariance_div_alpha_'
    if symmetrize_div:
        name += 'sym_'
    name += str(alpha)

    class M(ComplexGaussianIG):
        def dist(self, x1, x2):
            if symmetrize_div:
                if div_alpha_real_case:
                    return self.div_alpha_sym_real_case(x1, x2)
                else:
                    return self.div_alpha_sym(x1, x2)
            else:
                if div_alpha_real_case:
                    return self.div_alpha_real_case(x1, x2)
                else:
                    return self.div_alpha(x1, x2)

    args_M = {'sizes': p, 'alpha': alpha}

    return Feature(name, Gaussian_estimation, M, args_M,
                   min_grad_norm_mean=min_grad_norm_mean)
