import autograd.numpy as np
from numpy import testing as np_test

from pyCovariance.generation_data import generate_complex_covariance, \
        generate_complex_stiefel, \
        generate_covariance, \
        generate_stiefel, \
        generate_textures, \
        generate_toeplitz, \
        sample_complex_compound_distribution, \
        sample_complex_normal_distribution, \
        sample_complex_standard_normal_distribution, \
        sample_complex_tau_UUH_distribution, \
        sample_normal_distribution, \
        sample_standard_normal_distribution, \
        sample_tau_UUH_distribution


def test_generate_covariance():
    p = 5
    sigma = generate_covariance(p)

    # test if sigma is SPD
    assert sigma.dtype == np.float64
    np_test.assert_almost_equal(sigma, sigma.T, decimal=3)
    eigvals, _ = np.linalg.eigh(sigma)
    assert (eigvals > 0).all()

    sigma = generate_covariance(p, unit_det=True)

    # test if sigma is SPD
    assert sigma.dtype == np.float64
    np_test.assert_almost_equal(sigma, sigma.T, decimal=3)
    eigvals, _ = np.linalg.eigh(sigma)
    assert (eigvals > 0).all()

    # test unit det
    det = np.linalg.det(sigma)
    np_test.assert_almost_equal(det, 1)


def test_generate_complex_covariance():
    p = 5
    sigma = generate_complex_covariance(p)

    # test if sigma is HPD
    assert sigma.dtype == np.complex128
    np_test.assert_almost_equal(sigma, sigma.conj().T, decimal=3)
    eigvals, _ = np.linalg.eigh(sigma)
    assert (eigvals > 0).all()

    sigma = generate_complex_covariance(p, unit_det=True)

    # test if sigma is HPD
    assert sigma.dtype == np.complex128
    np_test.assert_almost_equal(sigma, sigma.conj().T, decimal=3)
    eigvals, _ = np.linalg.eigh(sigma)
    assert (eigvals > 0).all()

    # test unit det
    det = np.linalg.det(sigma)
    np_test.assert_almost_equal(det, 1)


def test_generate_toeplitz():
    rho = 0.8
    p = 5
    sigma = generate_toeplitz(p, rho)

    assert sigma.dtype == np.float64
    # test upper part of sigma
    for j in range(p):
        for i in range(p-j):
            np_test.assert_almost_equal(sigma[0, j], sigma[i, i+j])
    # test lower part of sigma
    sigma = sigma.T
    for j in range(p):
        for i in range(p-j):
            np_test.assert_almost_equal(sigma[0, j], sigma[i, i+j])


def test_generate_textures():
    N = 20
    textures = generate_textures(N)
    assert textures.dtype == np.float64
    for t in textures:
        assert t > 0


def test_generate_stiefel():
    p = 10
    k = 3
    U = generate_stiefel(p, k)
    assert U.dtype == np.float64
    np_test.assert_allclose(U.T@U, np.eye(k), atol=1e-10)


def test_generate_complex_stiefel():
    p = 10
    k = 3
    U = generate_complex_stiefel(p, k)
    assert U.dtype == np.complex128
    np_test.assert_allclose(U.conj().T@U, np.eye(k), atol=1e-10)


def test_sample_standard_normal_distribution():
    p = 5
    N = 10

    X = sample_standard_normal_distribution(p, N)
    assert X.dtype == np.float64
    assert X.shape == (p, N)
    # Other tests are carried out in the SCM tests.


def test_sample_complex_standard_normal_distribution():
    p = 5
    N = 10

    X = sample_complex_standard_normal_distribution(p, N)
    assert X.dtype == np.complex128
    assert X.shape == (p, N)
    # Other tests are carried out in the SCM tests.


def test_sample_normal_distribution():
    p = 5
    N = 10

    sigma = generate_covariance(p)
    X = sample_normal_distribution(N, sigma)
    assert X.dtype == np.float64
    assert X.shape == (p, N)
    # Other tests are carried out in the SCM tests.


def test_sample_complex_normal_distribution():
    p = 5
    N = 10

    sigma = generate_complex_covariance(p)
    X = sample_complex_normal_distribution(N, sigma)
    assert X.dtype == np.complex128
    assert X.shape == (p, N)
    # Other tests are carried out in the SCM tests.


def test_sample_complex_compound_distribution():
    p = 5
    N = 10

    sigma = generate_complex_covariance(p)
    sigma = sigma/(np.linalg.det(sigma)**(1/p))
    tau = generate_textures(N)
    X = sample_complex_compound_distribution(tau, sigma)
    assert X.dtype == np.complex128
    assert X.shape == (p, N)
    # Other tests are carried out in the Tyler tests.


def test_sample_tau_UUH_distribution():
    p = 10
    k = 3
    N = 20

    U = generate_stiefel(p, k)
    tau = generate_textures(N)
    X = sample_tau_UUH_distribution(tau, U)
    assert X.dtype == np.float64
    assert X.shape == (p, N)
    # Other tests are carried out in the low rank estimation tests.


def test_sample_complex_tau_UUH_distribution():
    p = 10
    k = 3
    N = 20

    U = generate_complex_stiefel(p, k)
    tau = generate_textures(N)
    X = sample_complex_tau_UUH_distribution(tau, U)
    assert X.dtype == np.complex128
    assert X.shape == (p, N)
    # Other tests are carried out in the low rank estimation tests.
