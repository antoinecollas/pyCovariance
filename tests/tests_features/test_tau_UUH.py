import autograd.numpy as np
import autograd.numpy.linalg as la
import numpy.testing as np_test

from pyCovariance.features.base import _FeatureArray
from pyCovariance.features.tau_UUH import tau_UUH
from pyCovariance.generation_data import generate_stiefel,\
        generate_textures,\
        sample_tau_UUH_distribution


def test_real_tau_UUH():
    p = 5
    k = 2
    N = int(1e6)

    feature = tau_UUH(N, p, k)

    # test estimation
    tau = generate_textures(N)
    U = generate_stiefel(p, k)
    X = sample_tau_UUH_distribution(tau, U)
    assert X.dtype == np.float64
    est = feature.estimation(X)
    tau_est = est.export()[0]
    U_est = est.export()[1]
    assert tau_est.shape == (N, 1)
    assert tau_est.dtype == np.float64
    assert U_est.shape == (p, k)
    assert U_est.dtype == np.float64

    sym_U_est = U_est@U_est.conj().T
    sym_U = U@U.conj().T
    error = la.norm(sym_U_est - sym_U) / la.norm(sym_U)
    assert error < 0.01

    # test distance
    tau1 = generate_textures(N)
    U1 = generate_stiefel(p, k)
    theta1 = _FeatureArray((N, 1), (p, k))
    theta1.append([tau1, U1])
    tau2 = generate_textures(N)
    U2 = generate_stiefel(p, k)
    theta2 = _FeatureArray((p, k), (N, 1))
    theta2.append([tau2, U2])
    d1 = feature.distance(theta1, theta2)
    assert d1.ndim == 0
    assert d1.dtype == np.float64
    # compute Riemannian log following Absil04
    temp = U2@np.linalg.inv(U1.conj().T@U2)-U1
    U, S, Vh = np.linalg.svd(temp, full_matrices=False)
    theta = np.diag(np.arctan(S))
    log_U1_U2 = U@theta@Vh
    d2 = np.sqrt(la.norm(log_U1_U2)**2 + la.norm(np.log(tau1)-np.log(tau2))**2)
    np_test.assert_almost_equal(d1, d2)

    # test mean
    N = int(1e2)
    N_mean = 10
    theta = _FeatureArray((N, 1), (p, k))
    for i in range(N_mean):
        tau = generate_textures(N)
        U = generate_stiefel(p, k)
        theta.append([tau, U])
    m = feature.mean(theta).export()
    assert m[0].dtype == np.float64
    assert m[1].dtype == np.float64

    tau = theta.export()[0]
    m_tau = np.prod(tau, axis=0)**(1/N_mean)
    assert la.norm(m[0]-m_tau)/la.norm(m_tau) < 1e-8

    grad = 0
    U = theta.export()[1]
    for i in range(N_mean):
        temp = U[i]@np.linalg.inv(m[1].conj().T@U[i])-m[1]
        Q, S, Vh = np.linalg.svd(temp, full_matrices=False)
        temp = np.diag(np.arctan(S))
        grad += Q@temp@Vh
    grad *= -(1/N_mean)
    grad_norm = la.norm(grad)
    assert grad_norm < feature._eps_grad
