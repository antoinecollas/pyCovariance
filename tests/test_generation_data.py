import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd

from pyCovariance.generation_data import generate_complex_covariance, \
        generate_complex_stiefel, \
        generate_covariance, \
        generate_stiefel, \
        generate_textures_gamma_dist, \
        generate_textures_lognormal_dist, \
        generate_toeplitz, \
        sample_complex_compound_distribution, \
        sample_complex_normal_distribution, \
        sample_complex_standard_normal_distribution, \
        sample_complex_tau_UUH_distribution, \
        sample_compound_distribution, \
        sample_normal_distribution, \
        sample_standard_normal_distribution, \
        sample_tau_UUH_distribution
from pyCovariance.testing import assert_allclose


def test_generate_covariance():
    rnd.seed(123)

    p = 5
    sigma = generate_covariance(p)

    # test if sigma is SPD
    assert sigma.dtype == np.float64
    assert_allclose(sigma, sigma.T)
    eigvals, _ = la.eigh(sigma)
    assert (eigvals > 0).all()

    sigma = generate_covariance(p, unit_det=True)

    # test if sigma is SPD
    assert sigma.dtype == np.float64
    assert_allclose(sigma, sigma.T)
    eigvals, _ = la.eigh(sigma)
    assert (eigvals > 0).all()

    # test unit det
    det = la.det(sigma)
    assert_allclose(det, 1)


def test_generate_complex_covariance():
    rnd.seed(123)

    p = 5
    sigma = generate_complex_covariance(p)

    # test if sigma is HPD
    assert sigma.dtype == np.complex128
    assert_allclose(sigma, sigma.conj().T)
    eigvals, _ = la.eigh(sigma)
    assert (eigvals > 0).all()

    sigma = generate_complex_covariance(p, unit_det=True)

    # test if sigma is HPD
    assert sigma.dtype == np.complex128
    assert_allclose(sigma, sigma.conj().T)
    eigvals, _ = la.eigh(sigma)
    assert (eigvals > 0).all()

    # test unit det
    det = la.det(sigma)
    assert_allclose(det, 1)


def test_generate_toeplitz():
    rnd.seed(123)

    rho = 0.8
    p = 5
    sigma = generate_toeplitz(p, rho)

    assert sigma.dtype == np.float64
    # test upper part of sigma
    for j in range(p):
        for i in range(p-j):
            assert_allclose(sigma[0, j], sigma[i, i+j])
    # test lower part of sigma
    sigma = sigma.T
    for j in range(p):
        for i in range(p-j):
            assert_allclose(sigma[0, j], sigma[i, i+j])


def test_generate_textures_gamma_dist():
    rnd.seed(123)

    N = int(1e6)

    tau = generate_textures_gamma_dist(N, nu=1)
    assert tau.shape == (N, 1)
    assert tau.dtype == np.float64
    assert (tau > 0).all()
    assert np.abs(np.mean(tau) - 1) < 1e-2
    assert np.abs(np.var(tau) - 1) < 1e-2

    tau = generate_textures_gamma_dist(N, nu=0.1)
    assert tau.shape == (N, 1)
    assert tau.dtype == np.float64
    assert (tau > 0).all()
    assert np.abs(np.mean(tau) - 1) < 1e-2
    assert np.abs(np.var(tau) - 10) < 1e-1

    N = int(1e4)

    tau = generate_textures_gamma_dist(N, nu=0.1, unit_prod=True)
    assert tau.shape == (N, 1)
    assert tau.dtype == np.float64
    assert (tau > 0).all()
    assert_allclose(np.prod(tau), 1)

    N = int(1e4)

    tau = generate_textures_gamma_dist(N, nu=0.1, min_value=1e-16)
    assert tau.shape == (N, 1)
    assert tau.dtype == np.float64
    assert (tau >= 1e-16).all()

    tau = generate_textures_gamma_dist(
        N, nu=0.1, min_value=1e-16, unit_prod=True)
    assert tau.shape == (N, 1)
    assert tau.dtype == np.float64
    assert (tau >= 1e-16).all()
    assert_allclose(np.exp(np.mean(np.log(tau))), 1)


def test_generate_textures_lognormal_dist():
    rnd.seed(123)

    N = int(1e6)

    tau = generate_textures_lognormal_dist(N, variance=1)
    assert tau.shape == (N, 1)
    assert tau.dtype == np.float64
    assert (tau > 0).all()
    assert np.abs(np.mean(tau) - 1) < 1e-2
    var = np.exp(1) - 1
    assert (np.abs(np.var(tau) - var) / var) < 1e-2

    tau = generate_textures_lognormal_dist(N, variance=5)
    assert tau.shape == (N, 1)
    assert tau.dtype == np.float64
    assert (tau > 0).all()
    assert np.abs(np.mean(tau) - 1) < 2*1e-2
    var = np.exp(5) - 1
    assert (np.abs(np.var(tau) - var) / var) < 1e-1

    tau = generate_textures_lognormal_dist(N, variance=5, min_value=1e-16)
    assert tau.shape == (N, 1)
    assert tau.dtype == np.float64
    assert (tau >= 1e-16).all()
    assert np.abs(np.mean(tau) - 1) < 0.05
    # var = np.exp(5) - 1
    # assert (np.abs(np.var(tau) - var) / var) < 0.5

    N = int(1e3)

    tau = generate_textures_lognormal_dist(N, variance=5,  unit_prod=True)
    assert tau.shape == (N, 1)
    assert tau.dtype == np.float64
    assert (tau > 0).all()
    assert_allclose(np.exp(np.mean(np.log(tau))), 1)

    tau = generate_textures_lognormal_dist(N, variance=5,  unit_prod=True)
    assert tau.shape == (N, 1)
    assert tau.dtype == np.float64
    assert (tau > 1e-12).all()
    assert_allclose(np.exp(np.mean(np.log(tau))), 1)


def test_generate_stiefel():
    rnd.seed(123)

    p = 10
    k = 3
    U = generate_stiefel(p, k)
    assert U.dtype == np.float64
    assert_allclose(U.T@U, np.eye(k), atol=1e-10)


def test_generate_complex_stiefel():
    rnd.seed(123)

    p = 10
    k = 3
    U = generate_complex_stiefel(p, k)
    assert U.dtype == np.complex128
    assert_allclose(U.conj().T@U, np.eye(k), atol=1e-10)


def test_sample_standard_normal_distribution():
    rnd.seed(123)

    p = 5
    N = 10

    X = sample_standard_normal_distribution(p, N)
    assert X.dtype == np.float64
    assert X.shape == (p, N)
    # Other tests are carried out in the SCM tests.


def test_sample_complex_standard_normal_distribution():
    rnd.seed(123)

    p = 5
    N = 10

    X = sample_complex_standard_normal_distribution(p, N)
    assert X.dtype == np.complex128
    assert X.shape == (p, N)
    # Other tests are carried out in the SCM tests.


def test_sample_normal_distribution():
    rnd.seed(123)

    p = 5
    N = 10

    sigma = generate_covariance(p)
    X = sample_normal_distribution(N, sigma)
    assert X.dtype == np.float64
    assert X.shape == (p, N)
    # Other tests are carried out in the SCM tests.


def test_sample_complex_normal_distribution():
    rnd.seed(123)

    p = 5
    N = 10

    sigma = generate_complex_covariance(p)
    X = sample_complex_normal_distribution(N, sigma)
    assert X.dtype == np.complex128
    assert X.shape == (p, N)
    # Other tests are carried out in the SCM tests.


def test_sample_compound_distribution():
    rnd.seed(123)

    p = 5
    N = 10

    sigma = generate_covariance(p, unit_det=True)
    tau = generate_textures_gamma_dist(N)
    X = sample_compound_distribution(tau, sigma)
    assert X.dtype == np.float64
    assert X.shape == (p, N)
    # TODO: Add other tests in the Tyler tests ??


def test_sample_complex_compound_distribution():
    rnd.seed(123)

    p = 5
    N = 10

    sigma = generate_complex_covariance(p, unit_det=True)
    tau = generate_textures_gamma_dist(N)
    X = sample_complex_compound_distribution(tau, sigma)
    assert X.dtype == np.complex128
    assert X.shape == (p, N)
    # Other tests are carried out in the Tyler tests.


def test_sample_tau_UUH_distribution():
    rnd.seed(123)

    p = 10
    k = 3
    N = 20

    U = generate_stiefel(p, k)
    tau = generate_textures_gamma_dist(N)
    X = sample_tau_UUH_distribution(tau, U)
    assert X.dtype == np.float64
    assert X.shape == (p, N)
    # Other tests are carried out in the low rank estimation tests.


def test_sample_complex_tau_UUH_distribution():
    rnd.seed(123)

    p = 10
    k = 3
    N = 20

    U = generate_complex_stiefel(p, k)
    tau = generate_textures_gamma_dist(N)
    X = sample_complex_tau_UUH_distribution(tau, U)
    assert X.dtype == np.complex128
    assert X.shape == (p, N)
    # Other tests are carried out in the low rank estimation tests.
