import autograd.numpy as np
from autograd.numpy import random
from numpy import testing as np_test
import os, sys, time

from pyCovariance.features.covariance import distance_covariance_Riemannian, mean_covariance_Riemannian, mean_covariance_Riemannian_with_whitening, vech_SCM
from pyCovariance.generation_data import generate_covariance, sample_complex_normal
from pyCovariance.vectorization import unvech, vech
from pyCovariance.matrix_operators import sqrtm, invsqrtm, logm, expm


# test estimation of covariance
def test_compute_feature_Covariance_texture():
    N = 1000
    p = 3
    
    # generate sigma
    sigma = generate_covariance(p)
    
    # sample X from Gaussian(0, sigma)
    X = sample_complex_normal(N, sigma)
 
    # estimate sigma from X
    sigma_est = vech_SCM(X)

    assert distance_covariance_Riemannian(vech(sigma), sigma_est) < 0.2


# test Riemannian geometry of covariance matrices
def test_distance_covariance_Riemannian():
    p = 3
    cov_0 = generate_covariance(p)
    cov_1 = generate_covariance(p)
 
    isqrt_cov_0 = invsqrtm(cov_0)
    dist_1 = np.linalg.norm(logm(isqrt_cov_0@cov_1@isqrt_cov_0))
 
    cov_0 = vech(cov_0).reshape((-1,))
    cov_1 = vech(cov_1).reshape((-1,))
    dist_2 = distance_covariance_Riemannian(cov_0, cov_1)

    np_test.assert_almost_equal(dist_1, dist_2, decimal=3)


def test_mean_covariance_Riemannian():
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
    mean_opt = unvech(mean_covariance_Riemannian(covs, params))
    
    np_test.assert_almost_equal(mean, mean_opt, decimal=3)
   
    # same test but with multi processing
    params = [1.0, 0.95, 1e-3, 100, True, 8]
    mean_opt = unvech(mean_covariance_Riemannian(covs, params))
    
    np_test.assert_almost_equal(mean, mean_opt, decimal=3)

    # test that the gradient descent reaches global the minimum
    n = 10
    covs = np.zeros((int(p*(p+1)/2), n), dtype=np.complex)
    for i in range(n):
        covs[:, i] = vech(generate_covariance(p))
    mean = unvech(mean_covariance_Riemannian(covs))

    # check that minus the gradient of the cost function has a null norm
    sqrt_mean = sqrtm(mean)
    isqrt_mean = invsqrtm(mean)
    minus_gradient = 0
    for i in range(n):
        minus_gradient += logm(isqrt_mean@unvech(covs[:, i])@isqrt_mean)
    minus_gradient = sqrt_mean@minus_gradient@sqrt_mean
    
    np_test.assert_almost_equal(np.linalg.norm(minus_gradient), 0, decimal=2)

def test_mean_covariance_Riemannian_with_whitening():
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
    mean_opt = unvech(mean_covariance_Riemannian_with_whitening(covs, params))
    
    np_test.assert_almost_equal(mean, mean_opt, decimal=3)
   
    # test that the gradient descent reaches global the minimum
    n = 10
    covs = np.zeros((int(p*(p+1)/2), n), dtype=np.complex)
    for i in range(n):
        covs[:, i] = vech(generate_covariance(p))
    mean = unvech(mean_covariance_Riemannian_with_whitening(covs))

    # check that minus the gradient of the cost function has a null norm
    sqrt_mean = sqrtm(mean)
    isqrt_mean = invsqrtm(mean)
    minus_gradient = 0
    for i in range(n):
        minus_gradient += logm(isqrt_mean@unvech(covs[:, i])@isqrt_mean)
    minus_gradient = sqrt_mean@minus_gradient@sqrt_mean
    
    np_test.assert_almost_equal(np.linalg.norm(minus_gradient), 0, decimal=2)
