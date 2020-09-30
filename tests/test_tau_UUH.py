from autograd import grad
import autograd.numpy as np
from autograd.numpy import random
import numpy.testing as np_test
import os, sys, time

from pyCovariance.features.tau_UUH import distance_grass, estimation_tau_UUH_gradient_descent, estimation_tau_UUH, estimation_tau_UUH_SCM
from pyCovariance.generation_data import generate_stiefel, generate_texture, sample_tau_UUH

 
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


def test_estimation_tau_UUH_gradient_descent():
    p = 15
    k = 3
    N = 1000
    alpha = 10

    U = generate_stiefel(p, k)
    tau = alpha*generate_texture(N)
    X = sample_tau_UUH(tau, U)

    U_est, _ = estimation_tau_UUH_gradient_descent(X, k, autodiff=False)

    assert distance_grass(U, U_est) < 0.1

    U_est, _ = estimation_tau_UUH_gradient_descent(X, k, autodiff=True)

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
