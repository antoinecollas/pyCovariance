import autograd.numpy as np
from autograd.numpy import random
from numpy import testing as np_test
import os, sys, time

from pyCovariance.generation_data import generate_covariance, sample_complex_normal, sample_complex_standard_normal
from pyCovariance.vectorization import vech
from pyCovariance.features.covariance import distance_covariance_Riemannian


def test_generate_covariance():
    p = 3
    sigma = generate_covariance(p)
    
    # test if sigma is SPD
    np_test.assert_almost_equal(sigma, sigma.conj().T, decimal=3)
    eigvals, _  = np.linalg.eigh(sigma)
    assert (eigvals > 0).all()


def test_sample_complex_standard_normal():
    p = 3
    N = 10000

    # 0 mean
    X = sample_complex_standard_normal(p, N)
    mean = np.mean(X, axis=1)
    assert np.linalg.norm(mean) < 0.2

    # identity covariance
    SCM = (1/N)*X@X.conj().T
    assert distance_covariance_Riemannian(vech(SCM), vech(np.eye(p))) < 0.2


def test_sample_complex_normal():
    p = 3
    N = 10000

    sigma = generate_covariance(p)

    # 0 mean
    X = sample_complex_normal(N, sigma)
    mean = np.mean(X, axis=1)
    assert np.linalg.norm(mean) < 0.2

    # sigma covariance
    SCM = (1/N)*X@X.conj().T
    assert distance_covariance_Riemannian(vech(SCM), vech(sigma)) < 0.2
