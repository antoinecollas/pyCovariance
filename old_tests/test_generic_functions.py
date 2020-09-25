import autograd.numpy as np
from autograd.numpy import random
from numpy import testing as np_test
import os, sys, time

from pyCovariance.utils import unvec, unvech, vec, vech
from pyCovariance.generation_data import generate_covariance


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
