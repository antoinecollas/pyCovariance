import autograd
import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd
from pymanopt.manifolds import HermitianPositiveDefinite

from pyCovariance.manifolds import ComplexGaussianIG
from pyCovariance.matrix_operators import invsqrtm, logm, sqrtm
from pyCovariance.features.base import _FeatureArray
from pyCovariance.features import\
        covariance,\
        covariance_div_alpha,\
        covariance_euclidean
from pyCovariance.generation_data import\
        generate_complex_covariance,\
        generate_covariance,\
        sample_complex_normal_distribution,\
        sample_complex_standard_normal_distribution,\
        sample_normal_distribution,\
        sample_standard_normal_distribution
from pyCovariance.testing import assert_allclose


def test_real_covariance():
    rnd.seed(123)

    p = 5
    N = int(1e6)
    N_mean = 10
    cov = covariance()(p, N)
    assert type(str(cov)) is str

    # test estimation 1
    X = sample_standard_normal_distribution(p, N)

    scm = cov.estimation(X).export()
    assert scm.dtype == np.float64
    assert la.norm(scm-np.eye(p))/la.norm(np.eye(p)) < 0.01

    # test estimation 2
    sigma = generate_covariance(p)
    X = sample_normal_distribution(N, sigma)

    scm = cov.estimation(X).export()
    assert scm.dtype == np.float64
    assert la.norm(scm-sigma)/la.norm(sigma) < 0.01

    # test distance
    sigma = _FeatureArray((p, p))
    sigma.append(generate_covariance(p))
    sigma.append(generate_covariance(p))

    sigma0_isqrtm = invsqrtm(sigma[0].export())
    prod = sigma0_isqrtm@sigma[1].export()@sigma0_isqrtm
    eigvals = la.eigvalsh(prod)
    d = np.sqrt(np.sum(np.log(eigvals)**2))
    assert_allclose(cov.distance(sigma[0], sigma[1]), d)

    # test log
    X = _FeatureArray((p, p))
    X.append(generate_covariance(p))
    Y = _FeatureArray((p, p))
    Y.append(generate_covariance(p))

    log_sigma = cov.log(X, Y)
    assert type(log_sigma) is _FeatureArray
    assert len(log_sigma) == 1
    log_sigma = log_sigma.export()
    assert log_sigma.dtype == np.float64
    assert log_sigma.shape == (p, p)
    X_sqrtm = sqrtm(X.export())
    X_isqrtm = invsqrtm(X.export())
    temp = X_isqrtm @ Y.export() @ X_isqrtm
    temp = logm(temp)
    desired_log_sigma = X_sqrtm @ temp @ X_sqrtm
    assert_allclose(log_sigma, desired_log_sigma)

    # test vectorized log
    log_sigma = cov.log(X, Y, vectorize=True)
    assert type(log_sigma) is np.ndarray
    assert log_sigma.dtype == np.float64
    assert log_sigma.shape == (1, p*(p+1)/2)
    desired_log_sigma = desired_log_sigma[np.triu_indices(p)]
    desired_log_sigma = desired_log_sigma[np.newaxis, ...]
    assert_allclose(log_sigma, desired_log_sigma)

    # test batch log and vectorized log
    batch_size = 10
    X = _FeatureArray((p, p))
    X.append(generate_covariance(p))
    Y = _FeatureArray((p, p))
    for _ in range(batch_size):
        Y.append(generate_covariance(p))

    log_sigma = cov.log(X, Y)
    assert type(log_sigma) is _FeatureArray
    assert len(log_sigma) == batch_size
    log_sigma = log_sigma.export()
    assert type(log_sigma) == np.ndarray
    assert log_sigma.dtype == np.float64
    assert log_sigma.shape[1:] == (p, p)

    log_sigma_vec = cov.log(X, Y, vectorize=True)
    assert type(log_sigma_vec) is np.ndarray
    assert log_sigma_vec.dtype == np.float64
    assert log_sigma_vec.shape == (batch_size, p*(p+1)/2)

    X_sqrtm = sqrtm(X.export())
    X_isqrtm = invsqrtm(X.export())
    for i in range(batch_size):
        temp = X_isqrtm @ Y.export()[i] @ X_isqrtm
        temp = logm(temp)
        desired_log_sigma = X_sqrtm @ temp @ X_sqrtm

        condition = la.norm(log_sigma[i]-desired_log_sigma)
        condition = condition / la.norm(desired_log_sigma)
        assert condition < 1e-3

        desired_log_sigma = desired_log_sigma[np.triu_indices(p)]
        condition = la.norm(log_sigma_vec[i]-desired_log_sigma)
        condition = condition / la.norm(desired_log_sigma)
        assert condition < 1e-3

    # test mean 1
    sigma = _FeatureArray((p, p))
    sigma.append(generate_covariance(p))
    sigma.append(generate_covariance(p))

    sigma0 = sigma[0].export()
    sigma0_sqrtm = sqrtm(sigma0)
    sigma0_isqrtm = invsqrtm(sigma0)
    sigma1 = sigma[1].export()
    temp = sqrtm(sigma0_isqrtm@sigma1@sigma0_isqrtm)
    m_closed_form = sigma0_sqrtm@temp@sigma0_sqrtm
    m = cov.mean(sigma).export()
    assert m.dtype == np.float64
    assert la.norm(m-m_closed_form)/la.norm(m_closed_form) < 1e-6

    # test mean 2
    sigma = _FeatureArray((p, p))
    for _ in range(N_mean):
        sigma.append(generate_covariance(p))

    m = cov.mean(sigma).export()
    assert m.dtype == np.float64
    condition = 0
    for i in range(N_mean):
        sigmai = sigma[i].export()
        sigmai_sqrtm = sqrtm(sigmai)
        sigmai_isqrtm = invsqrtm(sigmai)
        temp = logm(sigmai_isqrtm@m@sigmai_isqrtm)
        condition += sigmai_isqrtm@temp@sigmai_sqrtm
    condition = la.norm(condition)
    assert condition < 1e-3


def test_real_centered_covariance():
    rnd.seed(123)

    p = 5
    N = int(1e6)
    cov = covariance(assume_centered=False)(p, N)
    assert type(str(cov)) is str

    # test estimation 1
    sigma = generate_covariance(p)
    mu = rnd.randn(p, 1)
    X = sample_normal_distribution(N, sigma) + mu

    scm = cov.estimation(X).export()
    assert scm.dtype == np.float64
    assert la.norm(scm-sigma)/la.norm(sigma) < 0.01


def test_mean_single_covariance():
    rnd.seed(123)

    p = 5
    N = 10
    cov = covariance()(p, N)
    sigma = _FeatureArray((p, p))
    sigma.append(generate_covariance(p))
    m = cov.mean(sigma)
    assert (sigma.export() == m.export()).all()


def test_complex_covariance():
    rnd.seed(123)

    p = 5
    N = int(1e6)
    N_mean = 10
    cov = covariance()(p, N)
    assert type(str(cov)) is str

    # test estimation 1
    X = sample_complex_standard_normal_distribution(p, N)

    scm = cov.estimation(X).export()
    assert scm.dtype == np.complex128
    assert la.norm(scm-np.eye(p))/la.norm(np.eye(p)) < 0.01

    # test estimation 2
    sigma = generate_complex_covariance(p)
    X = sample_complex_normal_distribution(N, sigma)

    scm = cov.estimation(X).export()
    assert scm.dtype == np.complex128
    assert la.norm(scm-sigma)/la.norm(sigma) < 0.01

    # test distance
    sigma = _FeatureArray((p, p))
    sigma.append(generate_complex_covariance(p))
    sigma.append(generate_complex_covariance(p))

    sigma0_isqrtm = invsqrtm(sigma[0].export())
    prod = sigma0_isqrtm@sigma[1].export()@sigma0_isqrtm
    eigvals = la.eigvalsh(prod)
    d = np.sqrt(np.sum(np.log(eigvals)**2))
    assert_allclose(cov.distance(sigma[0], sigma[1]), d)

    # test log
    X = _FeatureArray((p, p))
    X.append(generate_complex_covariance(p))
    Y = _FeatureArray((p, p))
    Y.append(generate_complex_covariance(p))

    log_sigma = cov.log(X, Y)
    assert type(log_sigma) is _FeatureArray
    assert len(log_sigma) == 1
    log_sigma = log_sigma.export()
    assert log_sigma.dtype == np.complex128
    assert log_sigma.shape == (p, p)
    X_sqrtm = sqrtm(X.export())
    X_isqrtm = invsqrtm(X.export())
    temp = X_isqrtm @ Y.export() @ X_isqrtm
    temp = logm(temp)
    desired_log_sigma = X_sqrtm @ temp @ X_sqrtm
    condition = la.norm(log_sigma - desired_log_sigma)
    condition = condition / la.norm(desired_log_sigma)
    assert condition < 1e-3

    # test vectorized log
    log_sigma = cov.log(X, Y, vectorize=True)
    assert type(log_sigma) is np.ndarray
    assert log_sigma.dtype == np.complex128
    assert log_sigma.shape == (1, p*(p+1)/2)
    desired_log_sigma = desired_log_sigma[np.triu_indices(p)]
    desired_log_sigma = desired_log_sigma[np.newaxis, ...]
    condition = la.norm(log_sigma - desired_log_sigma)
    condition = condition / la.norm(desired_log_sigma)
    assert condition < 1e-3

    # test batch log and vectorized log
    batch_size = 10
    X = _FeatureArray((p, p))
    X.append(generate_complex_covariance(p))
    Y = _FeatureArray((p, p))
    for _ in range(batch_size):
        Y.append(generate_complex_covariance(p))

    log_sigma = cov.log(X, Y)
    assert type(log_sigma) is _FeatureArray
    assert len(log_sigma) == batch_size
    log_sigma = log_sigma.export()
    assert type(log_sigma) == np.ndarray
    assert log_sigma.dtype == np.complex128
    assert log_sigma.shape[1:] == (p, p)

    log_sigma_vec = cov.log(X, Y, vectorize=True)
    assert type(log_sigma_vec) is np.ndarray
    assert log_sigma_vec.dtype == np.complex128
    assert log_sigma_vec.shape == (batch_size, p*(p+1)/2)

    X_sqrtm = sqrtm(X.export())
    X_isqrtm = invsqrtm(X.export())
    for i in range(batch_size):
        temp = X_isqrtm @ Y.export()[i] @ X_isqrtm
        temp = logm(temp)
        desired_log_sigma = X_sqrtm @ temp @ X_sqrtm

        condition = la.norm(log_sigma[i]-desired_log_sigma)
        condition = condition / la.norm(desired_log_sigma)
        assert condition < 1e-8

        desired_log_sigma = desired_log_sigma[np.triu_indices(p)]
        condition = la.norm(log_sigma_vec[i]-desired_log_sigma)
        condition = condition / la.norm(desired_log_sigma)
        assert condition < 1e-8

    # test mean 1
    sigma = _FeatureArray((p, p))
    sigma.append(generate_complex_covariance(p))
    sigma.append(generate_complex_covariance(p))

    sigma0 = sigma[0].export()
    sigma0_sqrtm = sqrtm(sigma0)
    sigma0_isqrtm = invsqrtm(sigma0)
    sigma1 = sigma[1].export()
    temp = sqrtm(sigma0_isqrtm@sigma1@sigma0_isqrtm)
    m_closed_form = sigma0_sqrtm@temp@sigma0_sqrtm
    m = cov.mean(sigma).export()
    assert m.dtype == np.complex128
    assert la.norm(m-m_closed_form)/la.norm(m_closed_form) < 1e-6

    # test mean 2
    sigma = _FeatureArray((p, p))
    for _ in range(N_mean):
        sigma.append(generate_complex_covariance(p))

    m = cov.mean(sigma).export()
    assert m.dtype == np.complex128
    condition = 0
    for i in range(N_mean):
        sigmai = sigma[i].export()
        sigmai_sqrtm = sqrtm(sigmai)
        sigmai_isqrtm = invsqrtm(sigmai)
        temp = logm(sigmai_isqrtm@m@sigmai_isqrtm)
        condition += sigmai_isqrtm@temp@sigmai_sqrtm
    condition = la.norm(condition)
    assert condition < 1e-4


def test_real_covariance_euclidean():
    rnd.seed(123)

    p = 5
    N = int(1e6)
    N_mean = 10
    cov = covariance_euclidean()(p, N)
    assert type(str(cov)) is str

    # test estimation
    sigma = generate_covariance(p)
    X = sample_normal_distribution(N, sigma)

    scm = cov.estimation(X).export()
    assert scm.dtype == np.float64
    assert la.norm(scm-sigma)/la.norm(sigma) < 0.01

    # test distance
    sigma = _FeatureArray((p, p))
    sigma.append(generate_covariance(p))
    sigma.append(generate_covariance(p))

    d = la.norm(sigma[0].export() - sigma[1].export())
    assert_allclose(cov.distance(sigma[0], sigma[1]), d)

    # test mean
    sigma = _FeatureArray((p, p))
    for _ in range(N_mean):
        sigma.append(generate_covariance(p))

    m = np.mean(sigma.export(), axis=0)
    assert cov.mean(sigma).export().dtype == np.float64
    assert_allclose(cov.mean(sigma).export(), m)


def test_real_covariance_div_KL_0():
    rnd.seed(123)

    p = 5
    N = int(1e6)
    alpha = 0
    feature = covariance_div_alpha(
        assume_centered=True,
        alpha=alpha,
        div_alpha_real_case=True,
        symmetrize_div=False
    )(p, N)
    assert type(str(feature)) is str

    # test estimation
    sigma = generate_covariance(p)
    X = sample_normal_distribution(N, sigma)

    scm = feature.estimation(X).export()
    assert scm.dtype == np.float64
    assert la.norm(scm-sigma)/la.norm(sigma) < 0.01

    # test divergence
    theta = _FeatureArray((p, p))
    theta.append(generate_covariance(p))
    theta.append(generate_covariance(p))
    d = feature.distance(theta[0], theta[1])
    M = ComplexGaussianIG(p=p, alpha=alpha)
    p1 = [np.zeros((p, 1)), theta[0].export()]
    p2 = [np.zeros((p, 1)), theta[1].export()]
    d_true = np.sqrt(0.5*(M.div_KL(p1, p2)**2))
    assert type(d) is np.float64
    assert d == d_true

    # test mean
    N_mean = 100
    theta = _FeatureArray((p, p))
    for _ in range(N_mean):
        theta.append(generate_covariance(p))
    m = feature.mean(theta).export()
    assert m.dtype == np.float64
    assert m.shape == (p, p)

    def _cost(m):
        d = 0
        for i in range(N_mean):
            d = d + feature.distance(m, theta[i].export())**2
        return d/(2*N_mean)
    egrad = autograd.grad(_cost)(m)
    M = HermitianPositiveDefinite(p)
    rgrad = M.egrad2rgrad(m, egrad)
    assert M.norm(m, rgrad) < feature._min_grad_norm

    mean_true = np.zeros((p, p))
    for i in range(N_mean):
        mean_true += la.inv(theta[i].export())
    mean_true /= N_mean
    mean_true = la.inv(mean_true)
    assert_allclose(m, mean_true, rtol=1e-5, atol=1e-8)


def test_real_covariance_div_KL_1():
    rnd.seed(123)

    p = 5
    N = int(1e6)
    alpha = 1
    feature = covariance_div_alpha(
        assume_centered=True,
        alpha=alpha,
        div_alpha_real_case=True,
        symmetrize_div=False
    )(p, N)
    assert type(str(feature)) is str

    # test estimation
    sigma = generate_covariance(p)
    X = sample_normal_distribution(N, sigma)

    scm = feature.estimation(X).export()
    assert scm.dtype == np.float64
    assert la.norm(scm-sigma)/la.norm(sigma) < 0.01

    # test divergence
    theta = _FeatureArray((p, p))
    theta.append(generate_covariance(p))
    theta.append(generate_covariance(p))
    d = feature.distance(theta[0], theta[1])
    M = ComplexGaussianIG(p=p, alpha=alpha)
    p1 = [np.zeros((p, 1)), theta[0].export()]
    p2 = [np.zeros((p, 1)), theta[1].export()]
    d_true = np.sqrt(0.5*(M.div_KL(p2, p1)**2))
    assert type(d) is np.float64
    assert d == d_true

    # test mean
    N_mean = 100
    theta = _FeatureArray((p, p))
    for _ in range(N_mean):
        theta.append(generate_covariance(p))
    m = feature.mean(theta).export()
    assert m.dtype == np.float64
    assert m.shape == (p, p)

    def _cost(m):
        d = 0
        for i in range(N_mean):
            d = d + feature.distance(m, theta[i].export())**2
        return d/(2*N_mean)
    egrad = autograd.grad(_cost)(m)
    M = HermitianPositiveDefinite(p)
    rgrad = M.egrad2rgrad(m, egrad)
    assert M.norm(m, rgrad) < feature._min_grad_norm

    mean_true = np.zeros((p, p))
    for i in range(N_mean):
        mean_true += theta[i].export()
    mean_true /= N_mean
    assert_allclose(m, mean_true, rtol=1e-2, atol=1e-5)


def test_real_covariance_div_alpha_sym():
    rnd.seed(123)

    p = 5
    N = int(1e6)
    alpha = 0.33
    feature = covariance_div_alpha(
        assume_centered=True,
        alpha=alpha,
        div_alpha_real_case=True,
        symmetrize_div=True
    )(p, N)
    assert type(str(feature)) is str

    # test estimation
    sigma = generate_covariance(p)
    X = sample_normal_distribution(N, sigma)

    scm = feature.estimation(X).export()
    assert scm.dtype == np.float64
    assert la.norm(scm-sigma)/la.norm(sigma) < 0.01

    # test divergence
    theta = _FeatureArray((p, p))
    theta.append(generate_covariance(p))
    theta.append(generate_covariance(p))
    d = feature.distance(theta[0], theta[1])
    M = ComplexGaussianIG(p=p, alpha=alpha)
    p1 = [np.zeros((p, 1)), theta[0].export()]
    p2 = [np.zeros((p, 1)), theta[1].export()]
    d_true = M.div_alpha_sym_real_case(p1, p2)
    assert type(d) is np.float64
    assert d == d_true

    # test mean
    N_mean = 100
    theta = _FeatureArray((p, p))
    for _ in range(N_mean):
        theta.append(generate_covariance(p))
    m = feature.mean(theta).export()
    assert m.dtype == np.float64
    assert m.shape == (p, p)

    def _cost(m):
        d = 0
        for i in range(N_mean):
            d = d + feature.distance(m, theta[i].export())**2
        return d/(2*N_mean)
    egrad = autograd.grad(_cost)(m)
    M = HermitianPositiveDefinite(p)
    rgrad = M.egrad2rgrad(m, egrad)
    assert M.norm(m, rgrad) < feature._min_grad_norm
