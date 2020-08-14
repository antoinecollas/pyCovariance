import numpy as np
import scipy as sp
import warnings

from .covariance_clustering_functions import Riemannian_distance_covariance, Riemannian_mean_covariance
from .generic_functions import *

def compute_feature_Covariance_texture(𝐗, args):
    """ Serve to compute feature for Covariance and texture classificaiton.
        We use vech opeartion to save memory space on covariance.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N) array where p is the dimension of data and N the number
                    of samples used for estimation
            * args = (ϵ, iter_max) for Tyler estimator, where
                ** ϵ = tolerance for convergence
                ** iter_max = number of iterations max

        Outputs:
        ---------
            * 𝐱 = the feature for classification
        """

    ϵ, iter_max = args
    𝚺, δ, iteration = tyler_estimator_covariance_normalisedet(np.squeeze(𝐗), ϵ, iter_max)
    𝚺 = (𝚺+𝚺.conj().T)/2
    τ = np.diagonal(np.squeeze(𝐗).conj().T@np.linalg.inv(𝚺)@np.squeeze(𝐗))
    return list( np.hstack([vech(𝚺), τ]) )


def compute_feature_Covariance_texture_mean(𝐗, args):
    """ Serve to compute feature for Covariance and texture classificaiton.
        We use vech opeartion to save memory space on covariance. Texture is
        computed as the mean over the window.
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N) array where p is the dimension of data and N the number
                    of samples used for estimation
            * args = (ϵ, iter_max) for Tyler estimator, where
                ** ϵ = tolerance for convergence
                ** iter_max = number of iterations max

        Outputs:
        ---------
            * 𝐱 = the feature for classification
        """

    ϵ, iter_max = args
    𝚺, δ, iteration = tyler_estimator_covariance_normalisedet(np.squeeze(𝐗), ϵ, iter_max)
    τ = np.diagonal(np.squeeze(𝐗).conj().T@np.linalg.inv(𝚺)@np.squeeze(𝐗))
    return list( np.hstack([vech(𝚺), np.mean(τ)]) )


def Riemannian_distance_covariance_texture(𝐱_1, 𝐱_2, params=None):
    """ Riemannian distance on covariance + texture parameters
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐱_1 = a (p*(p+1)/2+N,) numpy array corresponding to the stack of vech 
                    of the covariance matrix and textures for sample 1
            * 𝐱_2 = a (p*(p+1)/2+N,) numpy array corresponding to the stack of vech 
                    of the covariance matrix and textures for sample 2
            * params = (p, N)
        Outputs:
        ---------
            * d = the distance between samples
        """
    p, N = params
    
    dist_cov = Riemannian_distance_covariance(𝐱_1[:int(p*(p+1)/2)],𝐱_2[:int(p*(p+1)/2)])
    
    τ_1 = 𝐱_1[int(p*(p+1)/2):]
    τ_2 = 𝐱_2[int(p*(p+1)/2):]
    dist_τ = np.linalg.norm(np.log(𝛕_1)-np.log(𝛕_2))

    d = np.sqrt((1/p)*(dist_cov**2)+(1/n)*(dist_τ**2))

    return np.real(d)


def Riemannian_mean_covariance_texture(𝐗_class, mean_parameters=None):
    """ Riemannian mean on covariance + texture manifold:
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐗_class = array of shape (p*(p+1)/2 + N, M) corresponding to 
                        samples in class
            * mean_parameters = (ϵ, ϵ_step, tol, iter_max, enable_multi, number_of_threads) where
                * ϵ_start controls the speed of the gradient descent at first step
                * ϵ_update is the step of in line search: ϵ = ϵ_start * ϵ_update at each descending step
                * tol is the tolerance to stop the gradient descent
                * iter_max is the maximum number of iteration
                * enable_multi is a boolean to activate parrallel computation
                * number_of_threads is the number of threas for parrallel computation

        Outputs:
        ---------
            * 𝛍 = the vech of Riemannian mean
        """

    p, N, ϵ, ϵ_step, tol, iter_max, enable_multi, number_of_threads = mean_parameters

    # Splitting covariance and texture features
    𝐗_𝚺 = 𝐗_class[:int(p*(p+1)/2),:]
    𝐗_𝛕 = 𝐗_class[int(p*(p+1)/2):,:]

    # Compuitng Riemannian mean on PDH set
    𝚺_mean = Riemannian_mean_covariance(𝐗_𝚺, (ϵ, ϵ_step, tol, iter_max, enable_multi, number_of_threads))

    # Computing geometric mean for textures
    𝛕_mean = np.exp( np.sum(np.log(𝐗_𝛕), axis=1) * (1.0/𝐗_𝛕.shape[1]) )

    # Staking and returning results
    return np.hstack([𝚺_mean, 𝛕_mean])


def Riemannian_distance_covariance_texture_old(𝐱_1, 𝐱_2, params=None):
    """ (old  but may be useful) Riemannian distance on covariance + 
        texture parameter
        ----------------------------------------------------------------------
        Inputs:
        --------
            * 𝐱_1 = a (p,) numpy array corresponding to the stack of vech 
                    of the covariance matrix and texture for sample 1
            * 𝐱_2 = a (p,) numpy array corresponding to the stack of vech 
                    of the covariance matrix and texture for sample 2
            * params = (α, β, tol, iter_max, scale) where
                    ** α = scale for independent terms
                    ** β = scale for crossed-terms
                    ** ϵ = tolerance for convergence
                    ** iter_max = number of iterations max
                    ** scale = either 'log' or 'linear' for distance scale
        Outputs:
        ---------
            * d = the distance between samples
        """
    α, β, tol, iter_max, scale = params
    𝚺_1 = unvech(𝐱_1[:-1])
    𝚺_2 = unvech(𝐱_2[:-1])
    τ_1 = 𝐱_1[-1]
    τ_2 = 𝐱_2[-1]
    i𝚺_1_sqm = np.linalg.inv(sp.linalg.sqrtm(𝚺_1))
    d = α * np.linalg.norm( np.log( i𝚺_1_sqm @ 𝚺_2 @ i𝚺_1_sqm ) ) + \
                α * np.sum( np.log( 𝛕_1 / 𝛕_2 )**2 ) + \
                β * np.sum( np.log( 𝛕_1 / 𝛕_2 ) )**2

    if scale=='log':
        return np.real(d)
    else:
        return np.exp(np.real(d))

