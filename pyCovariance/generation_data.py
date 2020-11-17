import autograd.numpy as np
from autograd.numpy import random
from scipy.linalg import toeplitz

from .matrix_operators import sqrtm


def generate_covariance(p):
    # Generate eigenvalues between 1 and 2
    # (eigenvalues of a symmetric matrix are always real).
    D = np.diag(np.ones((p)) + random.rand(p))

    # Generate an orthogonal matrix.
    Q, _ = np.linalg.qr(random.randn(p, p))

    sigma = Q@D@Q.T
    return sigma


def generate_complex_covariance(p):
    # Generate eigenvalues between 1 and 2
    # (eigenvalues of a symmetric matrix are always real).
    D = np.diag(np.ones((p)) + random.rand(p))

    # Generate an orthogonal matrix.
    Q, _ = np.linalg.qr(random.randn(p, p)+1j*random.randn(p, p))

    sigma = Q@D@Q.conj().T
    return sigma


def generate_toeplitz(p, rho):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * p = size of matrix
                * rho = a scalar
            Outputs:
                * the matrix """
    return toeplitz(np.power(rho, np.arange(0, p)))


def generate_textures(N):
    mu = 0.1
    texture = random.gamma(mu, 1/mu, size=(N, 1))
    return texture


def generate_stiefel(p, k):
    Q, _ = np.linalg.qr(random.randn(p, k))
    return Q


def generate_complex_stiefel(p, k):
    Q, _ = np.linalg.qr(random.randn(p, k)+1j*random.randn(p, k))
    return Q


def sample_standard_normal_distribution(p, N):
    return random.randn(p, N)


def sample_complex_standard_normal_distribution(p, N):
    X = random.randn(p, N) + 1j*random.randn(p, N)
    X = (1/np.sqrt(2))*X
    return X


def sample_normal_distribution(N, cov):
    assert cov.dtype == np.float64
    p = cov.shape[0]
    X = sample_standard_normal_distribution(p, N)
    X = sqrtm(cov)@X
    return X


def sample_complex_normal_distribution(N, cov):
    assert cov.dtype == np.complex128
    p = cov.shape[0]
    X = sample_complex_standard_normal_distribution(p, N)
    X = sqrtm(cov)@X
    return X


def sample_complex_compound_distribution(tau, cov):
    assert cov.dtype == np.complex128
    N = tau.shape[0]
    temp = np.sqrt(tau).reshape((1, -1))
    X = temp*sample_complex_normal_distribution(N, cov)
    return X


def sample_complex_tau_UUH_distribution(tau, U):
    assert U.dtype == np.complex128
    N = tau.shape[0]
    p, k = U.shape
    cn_k = sample_complex_normal_distribution(N, np.eye(k, dtype=np.complex128))
    cn_p = sample_complex_normal_distribution(N, np.eye(p, dtype=np.complex128))
    X = np.sqrt(tau).reshape((1, -1))*(U@cn_k) + cn_p
    return X
