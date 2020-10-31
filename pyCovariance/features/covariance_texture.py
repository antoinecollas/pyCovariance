import autograd.numpy as np
from pymanopt.manifolds import Product, SpecialHermitianPositiveDefinite, StrictlyPositiveVectors
import warnings

from .base import BaseClassFeatures

########## ESTIMATION ##########

def tyler_estimator_covariance(X, init=None, tol=0.001, iter_max=100):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * init = point on manifold to initialise estimation
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * tau
            * sigma
            * delta = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    p, N = X.shape
    if init is None:
        sigma = (1/N)*X@X.conj().T
        sigma = p*sigma/np.trace(sigma)
    else:
        _, sigma = init

    delta = np.inf # Distance between two iterations
    iteration = 0

    while (delta>tol) and (iteration<iter_max):
        # compute expression of Tyler estimator
        tau = (1/p) * np.real(np.einsum('ij,ji->i', np.conjugate(X).T@np.linalg.inv(sigma), X))
        X_bis = X / np.sqrt(tau)
        sigma_new = (1/N) * X_bis@X_bis.conj().T

        # condition for stopping
        delta = np.linalg.norm(sigma_new - sigma, 'fro') / np.linalg.norm(sigma, 'fro')
        iteration = iteration + 1

        # updating sigma
        sigma = sigma_new

    # imposing trace constraint: Tr(sigma) = p
    c = np.trace(sigma)/p
    sigma = sigma/c
    tau = c*tau

    if iteration == iter_max:
        warnings.warn('Estimation algorithm did not converge')

    tau = tau.reshape((-1, 1))

    return (tau, sigma, delta, iteration)


def tyler_estimator_covariance_normalisedet(X, init=None, tol=0.001, iter_max=100):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * init = point on manifold to initialise estimation
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * tau
            * sigma
            * delta = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    p, N = X.shape
    if init is None:
        sigma = (1/N)*X@X.conj().T
        sigma = sigma/(np.linalg.det(sigma)**(1/p))
    else:
        _, sigma = init

    delta = np.inf # Distance between two iterations
    iteration = 0

    while (delta>tol) and (iteration<iter_max):
        # compute expression of Tyler estimator
        tau = (1/p) * np.real(np.einsum('ij,ji->i', np.conjugate(X).T@np.linalg.inv(sigma), X))
        X_bis = X / np.sqrt(tau)
        sigma_new = (1/N) * X_bis@X_bis.conj().T

        # condition for stopping
        delta = np.linalg.norm(sigma_new - sigma, 'fro') / np.linalg.norm(sigma, 'fro')
        iteration = iteration + 1

        # updating sigma
        sigma = sigma_new

    # imposing det constraint: det(sigma) = 1
    c = np.linalg.det(sigma)**(1/p)
    sigma = sigma/c
    tau = c*tau

    if iteration == iter_max:
        warnings.warn('Estimation algorithm did not converge')

    tau = tau.reshape((-1, 1))

    return (tau, sigma, delta, iteration)


def compute_feature_covariance_texture(X, args=(0.001, 100)):
    """ Serve to compute feature for Covariance and texture classificaiton.
        We use vech opeartion to save memory space on covariance.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a (p, N) array where p is the dimension of data and N the number
                    of samples used for estimation
            * args = (ϵ, iter_max) for Tyler estimator, where
                ** eps = tolerance for convergence
                ** iter_max = number of iterations max

        Outputs:
        ---------
            * 𝐱 = the feature for classification
        """
    eps, iter_max = args
    tau, sigma, _, _ = tyler_estimator_covariance_normalisedet(np.squeeze(X), tol=eps, iter_max=iter_max)
    return tau, sigma


##########  CLASSES  ##########

class CovarianceTexture(BaseClassFeatures):
    def __init__(
        self,
        p,
        N,
        estimation_args=None,
    ):
        prod = Product([StrictlyPositiveVectors(N), SpecialHermitianPositiveDefinite(p)])
        super().__init__(manifold=prod)
        self.p = p
        self.N = N
        self.estimation_args = estimation_args

    def __str__(self):
        return 'Covariance_texture_Riemannian'

    def estimation(self, X):
        if self.estimation_args is not None:
            return compute_feature_covariance_texture(X, self.estimation_args)
        return compute_feature_covariance_texture(X)
