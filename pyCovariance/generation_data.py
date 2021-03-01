import autograd.numpy as np
import autograd.numpy.linalg as la
from autograd.numpy import random
from scipy.linalg import toeplitz
from scipy.stats import lognorm, ortho_group, unitary_group

from .matrix_operators import sqrtm


def generate_covariance(p, unit_det=False):
    # Generate eigenvalues as ch-square(1)
    # (eigenvalues of a symmetric matrix are always real).
    D = np.diag(random.rand(p)**2)

    # Generate an orthogonal matrix.
    Q = generate_stiefel(p, p)

    sigma = Q@D@Q.T

    if unit_det:
        sigma = sigma/(np.real(la.det(sigma))**(1/p))

    return sigma


def generate_complex_covariance(p, unit_det=False):
    # Generate eigenvalues between 1 and 2
    # (eigenvalues of a symmetric matrix are always real).
    D = np.diag(np.ones((p)) + random.rand(p))

    # Generate an orthogonal matrix.
    Q = generate_complex_stiefel(p, p)

    sigma = Q@D@Q.conj().T

    if unit_det:
        sigma = sigma/(np.real(la.det(sigma))**(1/p))

    return sigma


def generate_toeplitz(p, rho):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * p = size of matrix
                * rho = a scalar
            Outputs:
                * the matrix """
    return toeplitz(np.power(rho, np.arange(0, p)))


def generate_textures_gamma_dist(N, nu=0.1):
    textures = random.gamma(nu, 1/nu, size=(N, 1))
    return textures


def generate_textures_lognormal_dist(N, variance=10):
    s = np.sqrt(np.log(variance + 1))
    mu = -(s**2)/2
    textures = lognorm.rvs(scale=np.exp(mu), s=s, size=(N, 1))
    return textures


def generate_stiefel(p, k):
    if k < p:
        Q = ortho_group.rvs(p)[:, :k]
    else:
        Q = ortho_group.rvs(p)
    return Q


def generate_complex_stiefel(p, k):
    if k < p:
        Q = unitary_group.rvs(p)[:, :k]
    else:
        Q = unitary_group.rvs(p)
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


def sample_compound_distribution(tau, cov):
    assert cov.dtype == np.float64
    N = tau.shape[0]
    temp = np.sqrt(tau).reshape((1, -1))
    X = temp*sample_normal_distribution(N, cov)
    return X


def sample_complex_compound_distribution(tau, cov):
    assert cov.dtype == np.complex128
    N = tau.shape[0]
    temp = np.sqrt(tau).reshape((1, -1))
    X = temp*sample_complex_normal_distribution(N, cov)
    return X


def sample_tau_UUH_distribution(tau, U):
    assert U.dtype == np.float64
    N = tau.shape[0]
    p, k = U.shape
    cn_k = sample_normal_distribution(N, np.eye(k))
    cn_p = sample_normal_distribution(N, np.eye(p))
    X = np.sqrt(tau).reshape((1, -1))*(U@cn_k) + cn_p
    return X


def sample_complex_tau_UUH_distribution(tau, U):
    assert U.dtype == np.complex128
    N = tau.shape[0]
    p, k = U.shape
    cn_k = sample_complex_normal_distribution(N,
                                              np.eye(k, dtype=np.complex128))
    cn_p = sample_complex_normal_distribution(N,
                                              np.eye(p, dtype=np.complex128))
    X = np.sqrt(tau).reshape((1, -1))*(U@cn_k) + cn_p
    return X
