import autograd.numpy as np
import warnings

from .base import BaseClassFeatures
from .covariance import distance_covariance_Riemannian, mean_covariance_Riemannian
from ..vectorization import *

########## ESTIMATION ##########

def tyler_estimator_covariance(X, tol=0.001, iter_max=20):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * sigma
            * tau
            * delta = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = X.shape
    delta = np.inf # Distance between two iterations
    sigma = (1/N)*X@X.conj().T
    sigma = p*sigma/np.trace(sigma)
    iteration = 0

    # Recursive algorithm
    while (delta>tol) and (iteration<iter_max):
        # compute expression of Tyler estimator
        tau = np.real(np.einsum('ij,ji->i', np.conjugate(X).T@np.linalg.inv(sigma), X))
        X_bis = X / np.sqrt(tau)
        sigma_new = (1/N) * X_bis@X_bis.conj().T

        # imposing trace constraint: Tr(sigma) = p
        sigma_new = p*sigma_new/np.trace(sigma_new)

        # condition for stopping
        delta = np.linalg.norm(sigma_new - sigma, 'fro') / np.linalg.norm(sigma, 'fro')
        iteration = iteration + 1

        # updating sigma
        sigma = sigma_new

    if iteration == iter_max:
        warnings.warn('Recursive algorithm did not converge')

    return (tau, sigma, delta, iteration)


def tyler_estimator_covariance_normalisedet(X, tol=0.001, iter_max=20):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        and normalisation by determinant
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * sigma
            * tau
            * delta = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = X.shape
    delta = np.inf # Distance between two iterations
    sigma = (1/N)*X@X.conj().T
    sigma = sigma/(np.linalg.det(sigma)**(1/p))
    iteration = 0

    while (delta>tol) and (iteration<iter_max):
        # compute expression of Tyler estimator
        tau = np.real(np.einsum('ij,ji->i', np.conjugate(X).T@np.linalg.inv(sigma), X))
        X_bis = X / np.sqrt(tau)
        sigma_new = (1/N) * X_bis@X_bis.conj().T

        # imposing det constraint: det(sigma) = 1
        sigma_new = sigma_new/(np.linalg.det(sigma_new)**(1/p))

        # condition for stopping
        delta = np.linalg.norm(sigma_new - sigma, 'fro') / np.linalg.norm(sigma, 'fro')
        iteration = iteration + 1

        # updating sigma
        sigma = sigma_new
    
    if iteration == iter_max:
        warnings.warn('Recursive algorithm did not converge')

    return (tau, sigma, delta, iteration)


def compute_feature_covariance_texture(X, args=(0.01, 20)):
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
    tau, sigma, _, _ = tyler_estimator_covariance_normalisedet(np.squeeze(X), eps, iter_max)
    return np.hstack([vech(sigma), tau])


def compute_feature_covariance_texture_mean(X, args):
    """ Serve to compute feature for Covariance and texture classificaiton.
        We use vech opeartion to save memory space on covariance. Texture is
        computed as the mean over the window.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X = a (p, N) array where p is the dimension of data and N the number
                    of samples used for estimation
            * args = (eps, iter_max) for Tyler estimator, where
                ** eps = tolerance for convergence
                ** iter_max = number of iterations max

        Outputs:
        ---------
            * the feature for classification
        """

    eps, iter_max = args
    tau, sigma, _, _ = tyler_estimator_covariance_normalisedet(np.squeeze(X), eps, iter_max)
    return list(np.hstack([vech(sigma), np.mean(tau)]) )


##########  DISTANCE  ##########

def distance_texture_Riemannian(tau_1, tau_2):
    """ Riemannian distance on covariance + texture parameters
        ----------------------------------------------------------------------
        Inputs:
        --------
            * tau_1 = a (N,) numpy array corresponding to the textures
            * tau_2 = a (N,) numpy array corresponding to the textures
        Outputs:
        ---------
            * d = the distance between samples
        """
    return np.linalg.norm(np.log(tau_1)-np.log(tau_2))

def distance_covariance_texture_Riemannian(x_1, x_2, p, N):
    """ Riemannian distance on covariance + texture parameters
        ----------------------------------------------------------------------
        Inputs:
        --------
            * x_1 = a (p*(p+1)/2+N,) numpy array corresponding to the stack of vech 
                    of the covariance matrix and textures for sample 1
            * x_2 = a (p*(p+1)/2+N,) numpy array corresponding to the stack of vech
                    of the covariance matrix and textures for sample 2
            * p = size of covariance matrices
            * N = number of textures
        Outputs:
        ---------
            * d = the distance between samples
        """
    dist_cov = distance_covariance_Riemannian(x_1[:int(p*(p+1)/2)], x_2[:int(p*(p+1)/2)])
    dist_texture = distance_texture_Riemannian(x_1[int(p*(p+1)/2):], x_2[int(p*(p+1)/2):])

    d = np.sqrt((1/p)*(dist_cov**2)+(1/N)*(dist_texture**2))

    return d


##########   MEAN     ##########

def mean_covariance_texture_Riemannian(X_class, p, N, mean_parameters=[1.0, 0.95, 1e-3, 100, False, 0]):
    """ Riemannian mean on covariance + texture manifold:
        ----------------------------------------------------------------------
        Inputs:
        --------
            * X_class = array of shape (p*(p+1)/2 + N, M) corresponding to 
                        samples in class
            * p = size of covariance matrices
            * N = number of textures
            * mean_parameters = (eps, eps_step, tol, iter_max, enable_multi, number_of_threads) where
                * eps_start controls the speed of the gradient descent at first step
                * eps_update is the step of in line search: eps = eps_start * eps_update at each descending step
                * tol is the tolerance to stop the gradient descent
                * iter_max is the maximum number of iteration
                * enable_multi is a boolean to activate parrallel computation
                * number_of_threads is the number of threas for parrallel computation

        Outputs:
        ---------
            * mu = the vech of Riemannian mean
        """

    eps, eps_step, tol, iter_max, enable_multi, number_of_threads = mean_parameters

    # Splitting covariance and texture features
    X_sigma = X_class[:int(p*(p+1)/2),:]
    X_tau = X_class[int(p*(p+1)/2):,:]

    # Computing Riemannian mean on PDH set
    sigma_mean = mean_covariance_Riemannian(X_sigma, (eps, eps_step, tol, iter_max, enable_multi, number_of_threads))

    tau_mean = np.exp((1.0/X_tau.shape[1])*np.sum(np.log(X_tau), axis=1))

    # Staking and returning results
    return np.hstack([sigma_mean, tau_mean])


##########  CLASSES  ##########

class CovarianceTexture(BaseClassFeatures):
    def __init__(
        self,
        p,
        N,
        estimation_args=None,
        mean_args=None
    ):
        super().__init__()
        self.p = p
        self.N = N
        distance_args =  (p, N)
        self.estimation_args = estimation_args
        self.mean_args = mean_args
    
    def __str__(self):
        return 'Covariance_texture_Riemannian'
    
    def estimation(self, X):
        if self.estimation_args is not None:
            return compute_feature_covariance_texture(X, self.estimation_args)
        return compute_feature_covariance_texture(X)

    def distance(self, x1, x2):
        return distance_covariance_texture_Riemannian(x1, x2, self.p, self.N)

    def mean(self, X):
        if self.mean_args:
            return mean_covariance_texture_Riemannian(X, self.p, self.N, self.mean_args)
        return mean_covariance_texture_Riemannian(X, self.p, self.N)
