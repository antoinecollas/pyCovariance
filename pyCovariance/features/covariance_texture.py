import autograd.numpy as np
import autograd.numpy.linalg as la
from pymanopt.manifolds import\
        HermitianPositiveDefinite,\
        StrictlyPositiveVectors
import warnings

from .base import Feature
from ..matrix_operators import invsqrtm


# ESTIMATION


def tyler_estimator(X, init=None, tol=1e-8, iter_max=100):
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

    return tau, sigma, delta, iteration


def tyler_estimator_normalized_det(X, init=None, tol=1e-8, iter_max=100):
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

    tau, sigma, delta, iteration = tyler_estimator(X, init, tol, iter_max)

    # imposing det constraint: det(sigma) = 1
    c = np.real(la.det(sigma))**(1/p)
    sigma = sigma/c
    tau = c*tau

    return tau, sigma, delta, iteration


def tyler_estimator_normalized_trace(X, init=None, tol=1e-8, iter_max=100):
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

    tau, sigma, delta, iteration = tyler_estimator(X, init, tol, iter_max)

    # imposing trace constraint: Tr(sigma) = p
    c = np.real(np.trace(sigma))/p
    sigma = sigma/c
    tau = c*tau

    return tau, sigma, delta, iteration


# CLASSES


def covariance_texture(N, p, weights=None):
    name = 'Covariance_texture_Riemannian'
    # TODO: check why SpecialHermitianPositiveDefinite doesn't work ...
    M = (StrictlyPositiveVectors, HermitianPositiveDefinite)
    if weights is None:
        weights = (1/N, 1/p)
    args_M = {
        'sizes': (N, p),
        'weights': weights
    }

    def _tyler(X):
        tau, sigma, _, _ = tyler_estimator_normalized_det(X)
        return tau, sigma

    return Feature(name, _tyler, M, args_M)
