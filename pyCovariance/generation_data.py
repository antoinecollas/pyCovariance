import autograd.numpy as np
import autograd.numpy.linalg as la
from autograd.numpy import random as rnd
from scipy.linalg import toeplitz
from scipy.stats import lognorm, ortho_group, unitary_group
import warnings

from .matrix_operators import sqrtm


def generate_covariance(p, unit_det=False):
    # Generate eigenvalues as chi-square(1)
    # (eigenvalues of a symmetric matrix are always real).
    D = np.diag(rnd.normal(size=p)**2)

    # Generate an orthogonal matrix.
    Q = generate_stiefel(p, p)

    sigma = Q@D@Q.T

    if unit_det:
        sigma = sigma/(np.real(la.det(sigma))**(1/p))

    return sigma


def generate_complex_covariance(p, unit_det=False):
    # Generate eigenvalues as chi-square(1)
    # (eigenvalues of a symmetric matrix are always real).
    D = np.diag(rnd.normal(size=p)**2)

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


def _generate_textures_unit_prod(textures, min_value):
    c = np.exp(np.mean(np.log(textures)))
    textures = textures / c

    ITER_MAX = 100
    idx = np.zeros((len(textures), 1))
    j = 0
    while (np.sum(textures < min_value) > 0) and (j < ITER_MAX):
        mask = textures < min_value
        textures[mask] = min_value
        c = np.exp(np.mean(np.log(textures)))
        textures = textures / c
        idx = np.logical_or(mask, idx)

    if j == ITER_MAX:
        warnings.warn('Could not generate textures properly...')

    fraction_thres_text = np.sum(idx) / len(textures)

    return textures, fraction_thres_text


def generate_textures_gamma_dist(N, nu=0.1, min_value=0, unit_prod=False):
    textures = rnd.gamma(nu, 1/nu, size=(N, 1))

    if unit_prod:
        textures, fraction_thres_text = _generate_textures_unit_prod(
            textures, min_value)
    else:
        mask = textures < min_value
        fraction_thres_text = np.sum(mask) / N
        textures[mask] = min_value

    if (fraction_thres_text > 0.1) and (N >= 30):
        warnings.warn('More than 10% of textures have been thresholded...')

    return textures


def generate_textures_lognormal_dist(N, variance=2.4,
                                     min_value=0, unit_prod=False):
    # s = np.sqrt(np.log(variance + 1))
    s = np.sqrt(variance)
    mu = -(s**2)/2
    textures = lognorm.rvs(scale=np.exp(mu), s=s, size=(N, 1))

    if unit_prod:
        textures, fraction_thres_text = _generate_textures_unit_prod(
            textures, min_value)
    else:
        mask = textures < min_value
        fraction_thres_text = np.sum(mask) / N
        textures[mask] = min_value

    if fraction_thres_text > 0.05:
        warnings.warn('More than 5% of textures have been thresholded...')

    return textures


def generate_stiefel(p, k):
    if k < p:
        Q = ortho_group.rvs(p)[:, :k]
    else:
        Q = ortho_group.rvs(p)
    return Q


def generate_complex_stiefel(p, k):
    if p <= 1e3:
        if k < p:
            Q = unitary_group.rvs(p)[:, :k]
        else:
            Q = unitary_group.rvs(p)
    else:
        A = rnd.normal(size=(p, k)) + 1j*rnd.normal(size=(p, k))
        w, _, vh = la.svd(A, full_matrices=False)
        Q = w@vh
    return Q


def sample_standard_normal_distribution(p, N):
    return rnd.randn(p, N)


def sample_complex_standard_normal_distribution(p, N):
    X = rnd.randn(p, N) + 1j*rnd.randn(p, N)
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
    cn_k = sample_standard_normal_distribution(k, N)
    cn_p = sample_standard_normal_distribution(p, N)
    X = np.sqrt(tau).reshape((1, -1))*(U@cn_k) + cn_p
    return X


def sample_complex_tau_UUH_distribution(tau, U):
    assert U.dtype == np.complex128
    N = tau.shape[0]
    p, k = U.shape
    cn_k = sample_complex_standard_normal_distribution(k, N)
    cn_p = sample_complex_standard_normal_distribution(p, N)
    X = np.sqrt(tau).reshape((1, -1))*(U@cn_k) + cn_p
    return X
