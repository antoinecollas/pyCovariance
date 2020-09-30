from autograd import grad
import autograd.numpy as np
from autograd.numpy import random
import numpy.testing as np_test
import os, sys, time

from pyCovariance.features.location_covariance_texture import create_cost_egrad_location_covariance_texture
from pyCovariance.generation_data import generate_covariance, generate_texture, generate_Toeplitz, sample_compound


def test_cost_location_covariance_texture():
    p = 3
    N = 20

    # test cost function value when mu=0, tau=1 and sigma=I
    mu = np.zeros((p, 1), dtype=np.complex)
    tau = np.ones(N)
    sigma = np.eye(p)
    X = sample_compound(tau, sigma)
    cost, egrad = create_cost_egrad_location_covariance_texture(X)
    
    L = cost(mu, tau, sigma)
    L_true = np.tensordot(X, X.conj(), X.ndim)
    np_test.assert_almost_equal(L, L_true)

    # test cost function value
    mu = random.randn(p) + 1j*random.randn(p)
    tau = generate_texture(N)
    sigma = generate_covariance(p)
    X = sample_compound(tau, sigma)
    cost, _ = create_cost_egrad_location_covariance_texture(X)

    L = cost(mu, tau, sigma)
    L_true = 0
    sigma_inv = np.linalg.inv(sigma)
    for i in range(N):
        x = X[:, i]-mu
        Q = np.real(x.conj().T@sigma_inv@x)
        L_true = L_true + p*np.log(tau[i])+Q/tau[i]
    L_true = np.real(L_true)
    np_test.assert_allclose(L, L_true)


def test_egrad_location_covariance_texture():
    p = 3
    N = 20

    # test egrad
    mu = random.randn(p) + 1j*random.randn(p)
    tau = generate_texture(N)
    sigma = generate_covariance(p)
    X = sample_compound(tau, sigma)
    cost, grad_close_form = create_cost_egrad_location_covariance_texture(X, autodiff=False)
    
    grad_num = grad(cost, argnum=list(range(3)))

    gc = grad_close_form(mu, tau, sigma)
    gn = np.conjugate(grad_num(mu, tau, sigma))
    # test grad mu
    np_test.assert_allclose(gc[0], gn[0])
    # test grad tau
    np_test.assert_allclose(gc[1], gn[1])
    # test grad sigma
    np_test.assert_allclose(gc[2], gn[2])
