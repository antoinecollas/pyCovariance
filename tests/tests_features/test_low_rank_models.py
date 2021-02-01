import autograd.numpy as np
import autograd.numpy.linalg as la
import numpy.testing as np_test

from pyCovariance.features.base import _FeatureArray
from pyCovariance.features.low_rank_models import\
        estimate_tau_UUH,\
        estimate_tau_UUH_RGD
from pyCovariance.features import\
        subspace_SCM,\
        subspace_tau_UUH,\
        subspace_tau_UUH_RGD,\
        tau_UUH
from pyCovariance.generation_data import\
        generate_covariance,\
        generate_stiefel,\
        generate_textures,\
        sample_normal_distribution,\
        sample_tau_UUH_distribution


def test_real_subspace_SCM():
    p = 5
    k = 2
    N = int(1e2)

    feature = subspace_SCM(k)(p, N)

    # test estimation
    cov = generate_covariance(p)
    X = sample_normal_distribution(N, cov)
    SCM = (1/N) * X@X.T
    d, Q = la.eigh(SCM)
    Q = Q[:, -1:-k-1:-1]

    U = feature.estimation(X).export()
    assert U.shape == (p, k)
    assert U.dtype == np.float64

    assert la.norm(Q@Q.T - U@U.T) / la.norm(Q@Q.T) < 1e-8


def test_real_subspace_tau_UUH():
    p = 5
    k = 2
    N = int(1e6)

    feature = subspace_tau_UUH(k)(p, N)

    # test estimation
    tau = generate_textures(N)
    U = generate_stiefel(p, k)
    X = sample_tau_UUH_distribution(tau, U)
    assert X.dtype == np.float64
    U_est = feature.estimation(X).export()
    assert U_est.shape == (p, k)
    assert U_est.dtype == np.float64

    sym_U_est = U_est@U_est.T
    sym_U = U@U.T
    error = la.norm(sym_U_est - sym_U) / la.norm(sym_U)
    assert error < 0.01


def test_real_subspace_tau_UUH_RGD():
    p = 5
    k = 2
    N = int(1e4)

    feature = subspace_tau_UUH_RGD(k)(p, N)

    # test estimation
    tau = generate_textures(N)
    U = generate_stiefel(p, k)
    X = sample_tau_UUH_distribution(tau, U)
    assert X.dtype == np.float64
    U_est = feature.estimation(X).export()
    assert U_est.shape == (p, k)
    assert U_est.dtype == np.float64

    sym_U_est = U_est@U_est.conj().T
    sym_U = U@U.conj().T
    error = la.norm(sym_U_est - sym_U) / la.norm(sym_U)
    assert error < 0.05
    # increasing N should decrease the error but it is too slow...

    # test with autodiff
    feature = subspace_tau_UUH_RGD(k, autodiff=True)(p, N)

    U_est = feature.estimation(X).export()
    assert U_est.shape == (p, k)
    assert U_est.dtype == np.float64

    sym_U_est = U_est@U_est.conj().T
    sym_U = U@U.conj().T
    error = la.norm(sym_U_est - sym_U) / la.norm(sym_U)
    assert error < 0.05
    # increasing N should decrease the error but it is too slow...


def test_real_tau_UUH():
    p = 5
    k = 2
    N = int(1e6)

    feature = tau_UUH(k)(p, N)

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
    theta2 = _FeatureArray((N, 1), (p, k))
    theta2.append([tau2, U2])
    d1 = feature.distance(theta1, theta2)
    assert d1.ndim == 0
    assert d1.dtype == np.float64
    # compute Riemannian log following Absil04
    temp = U2@np.linalg.inv(U1.conj().T@U2)-U1
    U, S, Vh = np.linalg.svd(temp, full_matrices=False)
    theta = np.diag(np.arctan(S))
    log_U1_U2 = U@theta@Vh
    d2 = (1/k)*(la.norm(log_U1_U2)**2)
    d2 += (1/N)*(la.norm(np.log(tau1)-np.log(tau2))**2)
    d2 = np.sqrt(d2)
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

    grad_U = 0
    U = theta.export()[1]
    for i in range(N_mean):
        temp = U[i]@np.linalg.inv(m[1].conj().T@U[i])-m[1]
        Q, S, Vh = np.linalg.svd(temp, full_matrices=False)
        temp = np.diag(np.arctan(S))
        grad_U += Q@temp@Vh
    grad_U *= -(1/N_mean)
    temp_U = (1/k)*(la.norm(grad_U)**2)

    grad_tau = -(1/N_mean)*np.sum(m_tau*(np.log(tau)-np.log(m_tau)), axis=0)
    temp_tau = (1/N)*np.sum((1/m_tau)*grad_tau*(1/m_tau)*grad_tau)

    grad_norm = np.sqrt(temp_tau + temp_U)

    assert grad_norm < 1e-6


def test_real_estimation_tau_UUH():
    p = 15
    k = 3
    N = int(1e5)

    # data generation
    tau = generate_textures(N)
    U = generate_stiefel(p, k)
    X = sample_tau_UUH_distribution(tau, U)
    assert X.dtype == np.float64

    tau_BCD, U_BCD = estimate_tau_UUH(X, k)
    assert tau_BCD.shape == (N, 1)
    assert tau_BCD.dtype == np.float64
    assert U_BCD.shape == (p, k)
    assert U_BCD.dtype == np.float64
    error = la.norm(U_BCD@U_BCD.T - U@U.T) / la.norm(U@U.T)
    assert error < 0.05

    N = int(1e3)
    tau = generate_textures(N)
    U = generate_stiefel(p, k)
    X = sample_tau_UUH_distribution(tau, U)
    assert X.dtype == np.float64

    tau_BCD, U_BCD = estimate_tau_UUH(X, k)
    tau_RGD, U_RGD = estimate_tau_UUH_RGD(X, k)
    assert tau_RGD.shape == (N, 1)
    assert tau_RGD.dtype == np.float64
    assert U_RGD.shape == (p, k)
    assert U_RGD.dtype == np.float64

    delta_U = la.norm(U_BCD@U_BCD.T - U_RGD@U_RGD.T) / la.norm(U_BCD@U_BCD.T)
    _, theta, _ = la.svd(U_BCD.T@U_RGD)
    delta_tau = la.norm(tau_BCD - tau_RGD) / la.norm(tau_BCD)
    assert delta_U < 0.01
    assert delta_tau < 0.01
