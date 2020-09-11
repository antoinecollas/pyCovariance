import numpy as np
from numpy import random
import pytest
import scipy as sp
import os, sys, time

current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(current_dir)
sys.path.insert(1, temp)

from clustering_SAR.covariance_clustering_functions import distance_covariance_Riemannian
from clustering_SAR.covariance_and_texture_clustering_functions import compute_feature_covariance_texture, mean_covariance_texture_Riemannian
from clustering_SAR.generation_data import generate_covariance, generate_texture, generate_Toeplitz, sample_complex_normal, sample_compound
from clustering_SAR.generic_functions import unvech, vech
from clustering_SAR.matrix_operators import sqrtm, invsqrtm, logm, expm

###################################################
# test estimation of covariance + texture
###################################################
def test_compute_feature_covariance_texture():
    N = 1000
    p = 3
    
    # generate sigma
    sigma = generate_covariance(p)
    sigma = sigma / (np.linalg.det(sigma)**(1/p))
    np.testing.assert_almost_equal(np.linalg.det(sigma), 1, decimal=3)

    # generate tau
    tau = generate_texture(N)
    tau = tau.reshape(-1)
    # test if tau is strictly positive
    assert (tau > 0).all()
   
    # sample X from Compound(0, tau_i*sigma)
    X = sample_compound(tau, sigma)
 
    # estimate tau and sigma from X
    args_estimation = (-np.inf, 100)
    param_est = compute_feature_covariance_texture(X, args_estimation)
    sigma_est = unvech(param_est[:int(p*(p+1)/2)])
    tau_est = param_est[int(p*(p+1)/2):]
    assert distance_covariance_Riemannian(vech(sigma), vech(sigma_est)) < 0.2
    

###################################################
# test Riemannian geometry of covariance + texture
###################################################
def test_mean_covariance_texture_Riemannian():
    p = 3
    N = 25
    cov_0 = generate_covariance(p)
    tau_0 = generate_texture(N)
    cov_1 = generate_covariance(p)
    tau_1 = generate_texture(N)
    
    # close form of the mean between two covariances matrices
    sqrt_cov_0 = sqrtm(cov_0)
    isqrt_cov_0 = invsqrtm(cov_0)
    mean_sigma = sqrt_cov_0@sqrtm(isqrt_cov_0@cov_1@isqrt_cov_0)@sqrt_cov_0
    # close form of the mean between two strictly positive vectors
    sqrt_tau_0 = tau_0**(1/2)
    isqrt_tau_0 = tau_0**(-1/2)
    mean_tau = sqrt_tau_0*np.sqrt(isqrt_tau_0*tau_1*isqrt_tau_0)*sqrt_tau_0
    mean_tau = mean_tau.reshape((-1,))

    # mean computed by optimization
    cov_0 = vech(cov_0).reshape((-1,1))
    cov_1 = vech(cov_1).reshape((-1,1))
    temp_0 = np.concatenate([cov_0, tau_0], axis=0)
    temp_1 = np.concatenate([cov_1, tau_1], axis=0)
    features = np.concatenate([temp_0, temp_1], axis=1)
    params = [1.0, 0.95, 1e-3, 100, False, 0]
    mean_opt = mean_covariance_texture_Riemannian(features, p, N, params)

    mean_tau_opt = mean_opt[int(p*(p+1)/2):]
    mean_sigma_opt = unvech(mean_opt[:int(p*(p+1)/2)])

    np.testing.assert_almost_equal(mean_sigma, mean_sigma_opt, decimal=3)
    np.testing.assert_almost_equal(mean_tau, mean_tau_opt, decimal=3)
