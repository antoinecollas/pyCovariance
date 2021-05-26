import autograd.numpy as np
import autograd.numpy.linalg as la
from autograd.numpy import random as rnd
import numpy.testing as np_test

from pyCovariance.features import\
        location_covariance_texture_Gaussian,\
        location_covariance_texture_Tyler,\
        location_covariance_texture
from pyCovariance.features.location_covariance_texture import\
        create_cost_egrad_ehess_location_covariance_texture
from pyCovariance.generation_data import\
        generate_complex_covariance,\
        generate_covariance,\
        generate_textures_gamma_dist,\
        sample_complex_compound_distribution,\
        sample_complex_normal_distribution,\
        sample_compound_distribution,\
        sample_normal_distribution


def test_real_location_covariance_texture_Gaussian():
    rnd.seed(123)

    N = int(1e6)
    p = 5
    feature = location_covariance_texture_Gaussian()(p, N)

    mu = rnd.randn(p, 1)
    sigma = generate_covariance(p)
    X = sample_normal_distribution(N, sigma)
    X = X + mu
    assert X.dtype == np.float64

    res = feature.estimation(X).export()
    assert res[0].dtype == np.float64
    assert res[1].dtype == np.float64
    assert res[2].dtype == np.float64
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    assert la.norm(sigma - res[1]*res[2][0])/la.norm(sigma) < 0.01


def test_complex_location_covariance_texture_Gaussian():
    rnd.seed(123)

    N = int(1e6)
    p = 5
    feature = location_covariance_texture_Gaussian()(p, N)

    mu = rnd.randn(p, 1) + 1j*rnd.randn(p, 1)
    sigma = generate_complex_covariance(p)
    X = sample_complex_normal_distribution(N, sigma)
    X = X + mu
    assert X.dtype == np.complex128

    res = feature.estimation(X).export()
    assert res[0].dtype == np.complex128
    assert res[1].dtype == np.complex128
    assert res[2].dtype == np.float64
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    assert la.norm(sigma - res[1]*res[2][0])/la.norm(sigma) < 0.01


def test_real_location_covariance_texture_Tyler():
    rnd.seed(123)

    N = int(1e5)
    p = 5
    feature = location_covariance_texture_Tyler()(p, N)

    mu = rnd.randn(p, 1)
    sigma = generate_covariance(p, unit_det=True)
    tau = generate_textures_gamma_dist(N)
    X = sample_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.float64

    res = feature.estimation(X).export()
    assert res[0].dtype == np.float64
    assert res[1].dtype == np.float64
    assert res[2].dtype == np.float64
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.05


def test_complex_location_covariance_texture_Tyler():
    rnd.seed(123)

    N = int(1e5)
    p = 5
    feature = location_covariance_texture_Tyler()(p, N)

    mu = rnd.randn(p, 1) + 1j*rnd.randn(p, 1)
    sigma = generate_complex_covariance(p, unit_det=True)
    tau = generate_textures_gamma_dist(N)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.complex128

    res = feature.estimation(X).export()
    assert res[0].dtype == np.complex128
    assert res[1].dtype == np.complex128
    assert res[2].dtype == np.float64
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.05


def test_cost_location_covariance_texture():
    rnd.seed(123)

    p = 3
    N = 20

    # test cost function value when mu=0, tau=1 and sigma=I
    mu = np.zeros((p, 1), dtype=np.complex128)
    tau = np.ones(N)
    sigma = np.eye(p, dtype=np.complex128)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    cost, _, _ = create_cost_egrad_ehess_location_covariance_texture(X)

    L = cost(mu, sigma, tau)
    L_true = np.tensordot(X, X.conj(), X.ndim)
    np_test.assert_almost_equal(L, L_true)

    # test cost function value
    mu = rnd.randn(p, 1) + 1j*rnd.randn(p, 1)
    tau = generate_textures_gamma_dist(N)
    sigma = generate_complex_covariance(p, unit_det=True)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    cost, _, _ = create_cost_egrad_ehess_location_covariance_texture(X)

    L = cost(mu, sigma, tau)
    L_true = 0
    sigma_inv = la.inv(sigma)
    for i in range(N):
        x = X[:, i] - mu.reshape(-1)
        Q = np.real(x.conj().T@sigma_inv@x)
        L_true = L_true + p*np.log(tau[i])+Q/tau[i]
    L_true = np.real(L_true)
    np_test.assert_almost_equal(L, L_true)


def test_egrad_location_covariance_texture():
    rnd.seed(123)

    p = 3
    N = 20

    # test egrad
    mu = rnd.randn(p, 1) + 1j*rnd.randn(p, 1)
    tau = generate_textures_gamma_dist(N)
    sigma = generate_complex_covariance(p, unit_det=True)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    _, egrad, _ = create_cost_egrad_ehess_location_covariance_texture(
        X, autodiff=False)
    _, egrad_num, _ = create_cost_egrad_ehess_location_covariance_texture(
        X, autodiff=True)

    gc = egrad(mu, sigma, tau)
    gn = egrad_num(mu, sigma, tau)

    # test grad mu
    np_test.assert_allclose(gc[0], gn[0])
    # test grad sigma
    np_test.assert_allclose(gc[1], gn[1])
    # test grad tau
    np_test.assert_allclose(gc[2], gn[2])


def test_ehess_location_covariance_texture():
    rnd.seed(123)

    p = 3
    N = 20

    # test ehess
    mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
    tau = generate_textures_gamma_dist(N)
    sigma = generate_complex_covariance(p, unit_det=True)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    _, _, ehess = create_cost_egrad_ehess_location_covariance_texture(
        X,
        autodiff=False
    )
    _, _, ehess_num = create_cost_egrad_ehess_location_covariance_texture(
        X,
        autodiff=True
    )

    xi_mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
    xi_tau = rnd.normal(size=(N, 1))
    xi_sigma = rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p))
    xi_sigma = (xi_sigma + xi_sigma.conj().T) / 2

    hc = ehess(mu, sigma, tau, xi_mu, xi_sigma, xi_tau)
    hn = ehess_num(mu, sigma, tau, xi_mu, xi_sigma, xi_tau)

    # test ehess mu
    np_test.assert_allclose(hc[0], hn[0])
    # test ehess sigma
    np_test.assert_allclose(hc[1], hn[1])
    # test ehess tau
    np_test.assert_allclose(hc[2], hn[2])


def test_real_location_covariance_texture():
    rnd.seed(123)

    N = int(1e4)
    p = 3
    feature = location_covariance_texture(
        iter_max=1000,
        solver='conjugate'
    )(p, N)

    mu = rnd.randn(p, 1)
    sigma = generate_covariance(p, unit_det=True)
    tau = generate_textures_gamma_dist(N)
    X = sample_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.float64

    res = feature.estimation(X).export()
    assert res[0].dtype == np.float64
    assert res[1].dtype == np.float64
    assert res[2].dtype == np.float64
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.1


def test_complex_location_covariance_texture():
    rnd.seed(123)

    N = int(1e4)
    p = 3
    feature = location_covariance_texture(
        iter_max=200,
        solver='trust-regions'
    )(p, N)

    mu = rnd.randn(p, 1) + 1j*rnd.randn(p, 1)
    sigma = generate_complex_covariance(p, unit_det=True)
    tau = generate_textures_gamma_dist(N, nu=0.1)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.complex128

    res = feature.estimation(X).export()
    assert res[0].dtype == np.complex128
    assert res[1].dtype == np.complex128
    assert res[2].dtype == np.float64
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.05
