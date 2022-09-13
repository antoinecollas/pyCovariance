import autograd
import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd

from pyCovariance.features.base import _FeatureArray
from pyCovariance.features import\
        location_covariance_div_alpha,\
        location_covariance_orth_triangle,\
        location_covariance_scale_triangle
from pyCovariance.generation_data import\
        generate_covariance,\
        sample_normal_distribution
from pyCovariance.manifolds import ComplexGaussianIG


def test_real_location_covariance_div_alpha():
    rnd.seed(123)

    p = 5
    N = int(1e6)
    alpha = 0.33
    feature = location_covariance_div_alpha(
        alpha=alpha,
        div_alpha_real_case=True
    )(p, N)
    assert type(str(feature)) is str

    # test estimation
    sigma = generate_covariance(p)
    mu = rnd.normal(size=(p, 1))
    X = sample_normal_distribution(N, sigma) + mu

    res = feature.estimation(X).export()
    sample_mean, scm = res[0], res[1]
    assert sample_mean.dtype == np.float64
    assert scm.dtype == np.float64
    assert la.norm(sample_mean-mu)/la.norm(mu) < 0.01
    assert la.norm(scm-sigma)/la.norm(sigma) < 0.01

    # test divergence
    theta = _FeatureArray((p, 1), (p, p))
    theta.append((rnd.normal(size=(p, 1)), generate_covariance(p)))
    theta.append((rnd.normal(size=(p, 1)), generate_covariance(p)))
    d = feature.distance(theta[0], theta[1])
    M = ComplexGaussianIG(p=p, alpha=alpha)
    d_true = M.div_alpha_real_case(theta[0].export(), theta[1].export())
    assert type(d) is np.float64
    assert d == d_true


def test_real_location_covariance_div_alpha_sym():
    rnd.seed(123)

    p = 5
    N = int(1e6)
    alpha = 0.33
    feature = location_covariance_div_alpha(
        alpha=alpha,
        symmetrize_div=True,
        div_alpha_real_case=True
    )(p, N)
    assert type(str(feature)) is str

    # test estimation
    sigma = generate_covariance(p)
    mu = rnd.normal(size=(p, 1))
    X = sample_normal_distribution(N, sigma) + mu

    res = feature.estimation(X).export()
    sample_mean, scm = res[0], res[1]
    assert sample_mean.dtype == np.float64
    assert scm.dtype == np.float64
    assert la.norm(sample_mean-mu)/la.norm(mu) < 0.01
    assert la.norm(scm-sigma)/la.norm(sigma) < 0.01

    # test divergence
    theta = _FeatureArray((p, 1), (p, p))
    theta.append((rnd.normal(size=(p, 1)), generate_covariance(p)))
    theta.append((rnd.normal(size=(p, 1)), generate_covariance(p)))
    d = feature.distance(theta[0], theta[1])
    M = ComplexGaussianIG(p=p, alpha=alpha)
    d_true = M.div_alpha_sym_real_case(
        theta[0].export(), theta[1].export())
    assert type(d) is np.float64
    assert d == d_true


def test_real_location_covariance_orth_triangle():
    rnd.seed(123)

    p = 5
    N = int(1e6)
    feature = location_covariance_orth_triangle()(p, N)
    assert type(str(feature)) is str

    # test estimation
    sigma = generate_covariance(p)
    mu = rnd.normal(size=(p, 1))
    X = sample_normal_distribution(N, sigma) + mu

    res = feature.estimation(X).export()
    sample_mean, scm = res[0], res[1]
    assert sample_mean.dtype == np.float64
    assert scm.dtype == np.float64
    assert la.norm(sample_mean-mu)/la.norm(mu) < 0.01
    assert la.norm(scm-sigma)/la.norm(sigma) < 0.01

    # test divergence
    theta = _FeatureArray((p, 1), (p, p))
    theta.append((rnd.normal(size=(p, 1)), generate_covariance(p)))
    theta.append((rnd.normal(size=(p, 1)), generate_covariance(p)))
    d = feature.distance(theta[0], theta[1])
    M = ComplexGaussianIG(p=p)
    d_true = M.div_orth(theta[0].export(), theta[1].export())
    assert type(d) is np.float64
    assert d == d_true


def test_real_location_covariance_scale_triangle():
    rnd.seed(123)

    p = 5
    N = int(1e6)
    feature = location_covariance_scale_triangle()(p, N)
    assert type(str(feature)) is str

    # test estimation
    sigma = generate_covariance(p)
    mu = rnd.normal(size=(p, 1))
    X = sample_normal_distribution(N, sigma) + mu

    res = feature.estimation(X).export()
    sample_mean, scm = res[0], res[1]
    assert sample_mean.dtype == np.float64
    assert scm.dtype == np.float64
    assert la.norm(sample_mean-mu)/la.norm(mu) < 0.01
    assert la.norm(scm-sigma)/la.norm(sigma) < 0.01

    # test divergence
    theta = _FeatureArray((p, 1), (p, p))
    theta.append((rnd.normal(size=(p, 1)), generate_covariance(p)))
    theta.append((rnd.normal(size=(p, 1)), generate_covariance(p)))
    d = feature.distance(theta[0], theta[1])
    M = ComplexGaussianIG(p=p)
    d_true = M.div_scale(theta[0].export(), theta[1].export())
    assert type(d) is np.float64
    assert d == d_true


def test_real_location_covariance_div_alpha_mean():
    rnd.seed(123)

    p = 5
    N = 20
    N_mean = 100
    alpha = 0.33
    feature = location_covariance_div_alpha(
        alpha=alpha,
        div_alpha_real_case=True
    )(p, N)

    theta = _FeatureArray((p, 1), (p, p))
    for _ in range(N_mean):
        theta.append([
            rnd.normal(size=(p, 1)),
            generate_covariance(p)
        ])
    m = feature.mean(theta).export()
    assert len(m) == 2
    assert m[0].dtype == np.float64
    assert m[0].shape == (p, 1)
    assert m[1].dtype == np.float64
    assert m[1].shape == (p, p)

    def _cost(*m):
        d = 0
        for i in range(N_mean):
            d = d + feature.distance(m, theta[i].export())**2
        return d/(2*N_mean)
    egrad = list(autograd.grad(_cost, argnum=[0, 1])(*m))
    M = ComplexGaussianIG(p)
    rgrad = M.egrad2rgrad(m, egrad)
    assert M.norm(m, rgrad) < 1e-5


def test_real_location_covariance_div_alpha_sym_mean():
    rnd.seed(123)

    p = 5
    N = 20
    N_mean = 100
    alpha = 0.33
    feature = location_covariance_div_alpha(
        alpha=alpha,
        symmetrize_div=True,
        div_alpha_real_case=True
    )(p, N)

    theta = _FeatureArray((p, 1), (p, p))
    for _ in range(N_mean):
        theta.append([
            rnd.normal(size=(p, 1)),
            generate_covariance(p)
        ])
    m = feature.mean(theta).export()
    assert len(m) == 2
    assert m[0].dtype == np.float64
    assert m[0].shape == (p, 1)
    assert m[1].dtype == np.float64
    assert m[1].shape == (p, p)

    def _cost(*m):
        d = 0
        for i in range(N_mean):
            d = d + feature.distance(m, theta[i].export())**2
        return d/(2*N_mean)
    egrad = list(autograd.grad(_cost, argnum=[0, 1])(*m))
    M = ComplexGaussianIG(p, alpha=alpha)
    rgrad = M.egrad2rgrad(m, egrad)
    assert M.norm(m, rgrad) < 1e-5


def test_real_location_covariance_orth_triangle_mean():
    rnd.seed(123)

    p = 5
    N = 20
    N_mean = 100
    feature = location_covariance_orth_triangle()(p, N)

    theta = _FeatureArray((p, 1), (p, p))
    for _ in range(N_mean):
        theta.append([
            rnd.normal(size=(p, 1)),
            generate_covariance(p)
        ])
    m = feature.mean(theta).export()
    assert len(m) == 2
    assert m[0].dtype == np.float64
    assert m[0].shape == (p, 1)
    assert m[1].dtype == np.float64
    assert m[1].shape == (p, p)

    def _cost(*m):
        d = 0
        for i in range(N_mean):
            d = d + feature.distance(m, theta[i].export())**2
        return d/(2*N_mean)
    egrad = list(autograd.grad(_cost, argnum=[0, 1])(*m))
    M = ComplexGaussianIG(p)
    rgrad = M.egrad2rgrad(m, egrad)
    assert M.norm(m, rgrad) < 1e-5


def test_real_location_covariance_scale_triangle_mean():
    rnd.seed(123)

    p = 5
    N = 20
    N_mean = 3
    feature = location_covariance_scale_triangle()(p, N)

    theta = _FeatureArray((p, 1), (p, p))
    for _ in range(N_mean):
        theta.append([
            rnd.normal(size=(p, 1)),
            generate_covariance(p)
        ])
    m = feature.mean(theta).export()
    assert len(m) == 2
    assert m[0].dtype == np.float64
    assert m[0].shape == (p, 1)
    assert m[1].dtype == np.float64
    assert m[1].shape == (p, p)

    def _cost(*m):
        d = 0
        for i in range(N_mean):
            d = d + feature.distance(m, theta[i].export())**2
        return d/(2*N_mean)
    egrad = list(autograd.grad(_cost, argnum=[0, 1])(*m))
    M = ComplexGaussianIG(p)
    rgrad = M.egrad2rgrad(m, egrad)
    assert M.norm(m, rgrad) < 1e-5
