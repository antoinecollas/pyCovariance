import autograd.numpy as np
from autograd.numpy import random
from numpy import testing as np_test
import pytest
import os, sys, time

current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(current_dir)
sys.path.insert(1, temp)

from clustering_SAR.generic_functions import unvec, unvech, vec, vech
from clustering_SAR.generation_data import generate_covariance


def test_vec_unvec():
    p = 3
    sigma = generate_covariance(p)
    np_test.assert_almost_equal(sigma, unvec(vec(sigma)), decimal=3)
    
    p = 5
    sigma = generate_covariance(p)
    np_test.assert_almost_equal(sigma, unvec(vec(sigma)), decimal=3)
    
    p = 6
    sigma = generate_covariance(p)
    np_test.assert_almost_equal(sigma, unvec(vec(sigma)), decimal=3)

def test_vech_unvech():
    p = 3
    sigma = generate_covariance(p)
    np_test.assert_almost_equal(sigma, unvech(vech(sigma)), decimal=3)
    
    p = 5
    sigma = generate_covariance(p)
    np_test.assert_almost_equal(sigma, unvech(vech(sigma)), decimal=3)
    
    p = 6
    sigma = generate_covariance(p)
    np_test.assert_almost_equal(sigma, unvech(vech(sigma)), decimal=3)
