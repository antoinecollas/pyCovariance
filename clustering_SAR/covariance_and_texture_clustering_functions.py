import autograd.numpy as np
import warnings

from .covariance_clustering_functions import distance_covariance_Riemannian, mean_covariance_Riemannian
from .estimation import tyler_estimator_covariance_normalisedet
from .generic_functions import *

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


def compute_feature_Covariance_texture_mean(X, args):
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
