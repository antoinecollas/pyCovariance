import numpy as np
from numpy import random
import pytest
import os, sys, time

current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(current_dir)
sys.path.insert(1, temp)

from clustering_SAR.covariance_clustering_functions import Riemannian_distance_covariance, Riemannian_mean_covariance, vech_SCM
from clustering_SAR.generation_data import generate_covariance, sample_complex_normal
from clustering_SAR.generic_functions import unvech, vech
from clustering_SAR.matrix_operators import sqrtm, invsqrtm, logm, expm


###################################################
# test estimation of covariance
###################################################
def test_compute_feature_Covariance_texture():
    N = 1000
    p = 3
    
    # generate sigma
    sigma = generate_covariance(p)
    
    # sample X from Gaussian(0, sigma)
    X = sample_complex_normal(N, sigma)
 
    # estimate sigma from X
    sigma_est = vech_SCM(X)

    assert Riemannian_distance_covariance(vech(sigma), sigma_est) < 0.2


#################################################
# test Riemannian geometry of covariance matrices
#################################################
def test_Riemannian_distance_covariance():
    p = 3
    cov_0 = generate_covariance(p)
    cov_1 = generate_covariance(p)
 
    isqrt_cov_0 = invsqrtm(cov_0)
    dist_1 = np.linalg.norm(logm(isqrt_cov_0@cov_1@isqrt_cov_0))
 
    cov_0 = vech(cov_0).reshape((-1,))
    cov_1 = vech(cov_1).reshape((-1,))
    dist_2 = Riemannian_distance_covariance(cov_0, cov_1)

    np.testing.assert_almost_equal(dist_1, dist_2, decimal=3)


def test_Riemannian_mean_covariance():
    p = 3
    cov_0 = generate_covariance(p)
    cov_1 = generate_covariance(p)
    
    # close form of the mean between two covariances matrices
    sqrt_cov_0 = sqrtm(cov_0)
    isqrt_cov_0 = invsqrtm(cov_0)
    mean = sqrt_cov_0@sqrtm(isqrt_cov_0@cov_1@isqrt_cov_0)@sqrt_cov_0

    # mean computed by optimization
    cov_0 = vech(cov_0).reshape((-1,1))
    cov_1 = vech(cov_1).reshape((-1,1))
    covs = np.concatenate([cov_0, cov_1], axis=1)
    params = [1.0, 0.95, 1e-3, 100, False, 0]
    mean_opt = unvech(Riemannian_mean_covariance(covs, params))
    
    np.testing.assert_almost_equal(mean, mean_opt, decimal=3)
   
    # same test but with multi processing
    params = [1.0, 0.95, 1e-3, 100, True, 8]
    mean_opt = unvech(Riemannian_mean_covariance(covs, params))
    
    np.testing.assert_almost_equal(mean, mean_opt, decimal=3)
