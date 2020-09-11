import numpy as np
from numpy import random
import pytest
import os, sys, time

current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(current_dir)
sys.path.insert(1, temp)

from clustering_SAR.generation_data import generate_covariance, sample_complex_normal, sample_complex_standard_normal
from clustering_SAR.generic_functions import vech
from clustering_SAR.covariance_clustering_functions import distance_covariance_Riemannian


def test_generate_covariance():
    p = 3
    sigma = generate_covariance(p)
    
    # test if sigma is SPD
    np.testing.assert_almost_equal(sigma, sigma.conj().T, decimal=3)
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
