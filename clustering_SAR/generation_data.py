import numpy as np
from numpy import random
import scipy as sp

from .matrix_operators import sqrtm

def generate_covariance(p):
    N = 3*p
    X = random.standard_normal((p, N)) + 1j*random.standard_normal((p, N))
    cov = X@X.conj().T
    return cov

def generate_Toeplitz(rho, p):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * rho = a scalar
                * p = size of matrix
            Outputs:
                * the matrix """

    return sp.linalg.toeplitz(np.power(rho, np.arange(0, p)))

def generate_texture(N):
    mu = 0.1
    texture = random.gamma(mu, 1/mu, size=(N,1))
    return texture


def sample_complex_standard_normal(p, N):
    X = (1/np.sqrt(2))*(random.standard_normal((p,N)) + 1j*random.standard_normal((p,N)))
    return X


def sample_complex_normal(N, sigma):
    p = sigma.shape[0]
    X = sqrtm(sigma)@sample_complex_standard_normal(p, N)
    return X


def sample_compound(tau, sigma):
    N = tau.shape[0]
    p = sigma.shape[0]
    X = np.sqrt(tau).reshape((1, -1))*sample_complex_normal(N, sigma)
    return X
