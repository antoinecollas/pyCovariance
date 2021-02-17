import autograd.numpy as np
import autograd.numpy.linalg as la
from pymanopt.manifolds import\
        SpecialHermitianPositiveDefinite,\
        StrictlyPositiveVectors
import warnings

from .base import Feature, make_feature_prototype
from ..matrix_operators import invsqrtm


# ESTIMATION


def tyler_estimator(X, init=None, tol=1e-4, iter_max=100):
    """ A function that computes the Tyler Fixed Point Estimator
        for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N
            * init = point on manifold to initialise estimation
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * tau
            * sigma
            * delta = the final distance between two iterations
            * iteration = number of iterations until convergence """

    # Initialisation
    p, N = X.shape
    if init is None:
        sigma = (1/N)*X@X.conj().T
        sigma = p*sigma/np.trace(sigma)
    else:
        _, sigma = init

    delta = np.inf  # Distance between two iterations
    iteration = 0

    while (delta > tol) and (iteration < iter_max):
        # compute expression of Tyler estimator
        temp = invsqrtm(sigma)@X
        tau = np.einsum('ij,ji->i', temp.conj().T, temp)
        tau = (1/p) * np.real(tau)
        temp = X / np.sqrt(tau)
        sigma_new = (1/N) * temp@temp.conj().T

        # condition for stopping
        delta = la.norm(sigma_new - sigma) / la.norm(sigma)
        iteration = iteration + 1

        # updating sigma
        sigma = sigma_new

    if iteration == iter_max:
        warnings.warn('Estimation algorithm did not converge')

    tau = tau.reshape((-1, 1))

    return sigma, tau, delta, iteration


def tyler_estimator_normalized_det(X, init=None, tol=1e-4, iter_max=100):
    """ A function that computes the Tyler Fixed Point Estimator.
        Sigma is normalized to have a unit determinant.
        Inputs:
            * X = a matrix of size p*N
            * init = point on manifold to initialise estimation
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * tau
            * sigma
            * delta = the final distance between two iterations
            * iteration = number of iterations until convergence """

    p, _ = X.shape

    sigma, tau, delta, iteration = tyler_estimator(X, init, tol, iter_max)

    # imposing det constraint: det(sigma) = 1
    c = np.real(la.det(sigma))**(1/p)
    sigma = sigma/c
    tau = c*tau

    return sigma, tau, delta, iteration


def tyler_estimator_normalized_trace(X, init=None, tol=1e-4, iter_max=100):
    """ A function that computes the Tyler Fixed Point Estimator.
        Sigma is normalized to have tr(Sigma) = p
        Inputs:
            * X = a matrix of size p*N
            * init = point on manifold to initialise estimation
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * tau
            * sigma
            * delta = the final distance between two iterations
            * iteration = number of iterations until convergence """

    p, _ = X.shape

    sigma, tau, delta, iteration = tyler_estimator(X, init, tol, iter_max)

    # imposing trace constraint: Tr(sigma) = p
    c = np.real(np.trace(sigma))/p
    sigma = sigma/c
    tau = c*tau

    return sigma, tau, delta, iteration


# CLASSES


@make_feature_prototype
def covariance_texture(weights=None, p=None, N=None, **kwargs):
    M = (SpecialHermitianPositiveDefinite, StrictlyPositiveVectors)

    if weights is None:
        name = 'Covariance_texture'
    else:
        name = 'Covariance_' + str(round(weights[0], 2)) +\
               '_texture_' + str(round(weights[1], 2))

    if weights is None:
        weights = (1/p, 1/N)

    args_M = {
        'sizes': (p, N),
        'weights': weights
    }

    def _tyler(X):
        sigma, tau, _, _ = tyler_estimator_normalized_det(X)
        return sigma, tau

    return Feature(name, _tyler, M, args_M)
