from autograd import grad
import autograd.numpy as np
from autograd.numpy import random
import numpy.testing as np_test
import os, sys, time

from pyCovariance.features.tau_UUH import distance_grass, estimation_tau_UUH_RGD, estimation_tau_UUH, estimation_tau_UUH_SCM
from pyCovariance.generation_data import generate_stiefel, generate_texture, sample_tau_UUH

def test_distance_grass():
    p = 15
    k = 3

    U1 = generate_stiefel(p, k)
    U2 = generate_stiefel(p, k)

    # compute Riemannian log following Absil04
    U, S, Vh = np.linalg.svd(U2@np.linalg.inv(U1.conj().T@U2)-U1, full_matrices=False)
    theta = np.diag(np.arctan(S))
    log_U1_U2 = U@theta@Vh

    np_test.assert_almost_equal(np.linalg.norm(log_U1_U2), distance_grass(U1, U2))

 
def test_estimation_tau_UUH():
    p = 15
    k = 3
    N = 1000
    alpha = 10

    U = generate_stiefel(p, k)
    tau = alpha*generate_texture(N)
    X = sample_tau_UUH(tau, U)

    U_est, _, _, _ = estimation_tau_UUH(X, k, iter_max=100)

    assert distance_grass(U, U_est) < 0.1


def test_estimation_tau_UUH_RGD():
    p = 15
    k = 3
    N = 1000
    alpha = 10

    U = generate_stiefel(p, k)
    tau = alpha*generate_texture(N)
    X = sample_tau_UUH(tau, U)

    U_est, _ = estimation_tau_UUH_RGD(X, k, autodiff=False)

    assert distance_grass(U, U_est) < 0.1

    U_est, _ = estimation_tau_UUH_RGD(X, k, autodiff=True)

    assert distance_grass(U, U_est) < 0.1


def test_estimation_tau_UUH_SCM():
    p = 15
    k = 3
    N = 1000
    alpha = 10

    U = generate_stiefel(p, k)
    tau = alpha*generate_texture(N)
    X = sample_tau_UUH(tau, U)

    U_est, _, = estimation_tau_UUH_SCM(X, k)

    assert distance_grass(U, U_est) < 0.1
