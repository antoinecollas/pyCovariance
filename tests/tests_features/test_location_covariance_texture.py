from autograd import grad
import autograd.numpy as np
import autograd.numpy.linalg as la
from autograd.numpy import random
import numpy.testing as np_test

from pyCovariance.features import location_covariance_texture
from pyCovariance.features.location_covariance_texture import\
        create_cost_egrad_location_covariance_texture
from pyCovariance.generation_data import\
        generate_complex_covariance,\
        generate_covariance,\
        generate_textures,\
        sample_complex_compound_distribution,\
        sample_compound_distribution


def test_cost_location_covariance_texture():
    p = 3
    N = 20

    # test cost function value when mu=0, tau=1 and sigma=I
    mu = np.zeros((p, 1), dtype=np.complex128)
    tau = np.ones(N)
    sigma = np.eye(p, dtype=np.complex128)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    cost, _ = create_cost_egrad_location_covariance_texture(X)

    L = cost(mu, tau, sigma)
    L_true = np.tensordot(X, X.conj(), X.ndim)
    np_test.assert_almost_equal(L, L_true)

    # test cost function value
    mu = random.randn(p, 1) + 1j*random.randn(p, 1)
    tau = generate_textures(N)
    sigma = generate_complex_covariance(p, unit_det=True)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    cost, _ = create_cost_egrad_location_covariance_texture(X)

    L = cost(mu, tau, sigma)
    L_true = 0
    sigma_inv = la.inv(sigma)
    for i in range(N):
        x = X[:, i] - mu.reshape(-1)
        Q = np.real(x.conj().T@sigma_inv@x)
        L_true = L_true + p*np.log(tau[i])+Q/tau[i]
    L_true = np.real(L_true)
    np_test.assert_almost_equal(L, L_true)


def test_egrad_location_covariance_texture():
    p = 3
    N = 20

    # test egrad
    mu = random.randn(p, 1) + 1j*random.randn(p, 1)
    tau = generate_textures(N)
    sigma = generate_complex_covariance(p, unit_det=True)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    cost, grad_close_form = create_cost_egrad_location_covariance_texture(
        X, autodiff=False)

    grad_num = grad(cost, argnum=list(range(3)))

    gc = grad_close_form(mu, tau, sigma)
    gn = np.conjugate(grad_num(mu, tau, sigma))
    # test grad mu
    np_test.assert_allclose(gc[0], gn[0].reshape(-1))
    # test grad tau
    np_test.assert_allclose(gc[1], gn[1])
    # test grad sigma
    np_test.assert_allclose(gc[2], gn[2])


def test_real_location_covariance_texture():
    N = int(1e2)
    p = 5
    feature = location_covariance_texture(N, p)

    mu = random.randn(p, 1)
    tau = generate_textures(N)
    sigma = generate_covariance(p, unit_det=True)
    X = sample_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.float64

    res = feature.estimation(X).export()
    assert res[0].dtype == np.float64
    assert res[1].dtype == np.float64
    assert res[2].dtype == np.float64
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    # assert la.norm(sigma - res[2])/la.norm(sigma) < 0.01


def test_complex_location_covariance_texture():
    N = int(1e2)
    p = 5
    feature = location_covariance_texture(N, p)

    mu = random.randn(p, 1) + 1j*random.randn(p, 1)
    tau = generate_textures(N)
    sigma = generate_complex_covariance(p, unit_det=True)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.complex128

    res = feature.estimation(X).export()
    assert res[0].dtype == np.complex128
    assert res[1].dtype == np.float64
    assert res[2].dtype == np.complex128
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    # assert la.norm(sigma - res[2])/la.norm(sigma) < 0.01
