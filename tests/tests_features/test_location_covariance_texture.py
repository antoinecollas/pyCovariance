import autograd
import autograd.numpy as np
import autograd.numpy.linalg as la
from autograd.numpy import random as rnd

from pyCovariance.features.base import _FeatureArray
from pyCovariance.features import\
        location_covariance_orth_triangle,\
        location_covariance_texture_Gaussian_constrained_scatter,\
        location_covariance_texture_Gaussian_constrained_texture,\
        location_covariance_texture_Tyler_constrained_scatter,\
        location_covariance_texture_constrained_scatter,\
        location_covariance_texture_constrained_texture,\
        location_covariance_texture_constrained_texture_triangle,\
        location_covariance_texture_constrained_texture_div_alpha
from pyCovariance.features.location_covariance_texture import\
        create_cost_egrad_ehess_nll_CG_constrained_scatter,\
        create_cost_egrad_ehess_nll_CG_constrained_texture
from pyCovariance.generation_data import\
        generate_complex_covariance,\
        generate_covariance,\
        generate_textures_gamma_dist,\
        sample_complex_compound_distribution,\
        sample_complex_normal_distribution,\
        sample_compound_distribution,\
        sample_normal_distribution
from pyCovariance.manifolds import\
        ComplexGaussianIG,\
        ComplexCompoundGaussianIGConstrainedTexture
from pyCovariance.manifolds.product import _ProductTangentVector
from pyCovariance.testing import assert_allclose

# TEST OF FUNCTIONS WITH A CONSTRAINT OF UNITARY DETERMINANT
# ON THE SCATTER MATRIX


def test_real_location_covariance_texture_Gaussian_constrained_scatter():
    rnd.seed(123)

    N = int(1e6)
    p = 5
    feature = location_covariance_texture_Gaussian_constrained_scatter()(p, N)

    mu = rnd.normal(size=(p, 1))
    sigma = generate_covariance(p, unit_det=True)
    X = sample_normal_distribution(N, sigma)
    X = X + mu
    assert X.dtype == np.float64

    res = feature.estimation(X).export()
    assert res[0].dtype == np.float64
    assert res[1].dtype == np.float64
    assert res[2].dtype == np.float64
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    assert la.norm(sigma - res[1]*res[2][0])/la.norm(sigma) < 0.01


def test_complex_location_covariance_texture_Gaussian_constrained_scatter():
    rnd.seed(123)

    N = int(1e6)
    p = 5
    feature = location_covariance_texture_Gaussian_constrained_scatter()(p, N)

    mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
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


def test_real_location_covariance_texture_Tyler_constrained_scatter():
    rnd.seed(123)

    N = int(1e5)
    p = 5
    feature = location_covariance_texture_Tyler_constrained_scatter()(p, N)

    mu = rnd.normal(size=(p, 1))
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


def test_complex_location_covariance_texture_Tyler_constrained_scatter():
    rnd.seed(123)

    N = int(1e5)
    p = 5
    feature = location_covariance_texture_Tyler_constrained_scatter()(p, N)

    mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
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


def test_cost_location_covariance_texture_constrained_scatter():
    rnd.seed(123)

    p = 3
    N = 20

    # test cost function value when mu=0, tau=1 and sigma=I
    mu = np.zeros((p, 1), dtype=np.complex128)
    tau = np.ones(N)
    sigma = np.eye(p, dtype=np.complex128)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    cost, _, _ = create_cost_egrad_ehess_nll_CG_constrained_scatter(X)

    L = cost(mu, sigma, tau)
    L_true = np.tensordot(X, X.conj(), X.ndim)
    assert_allclose(L, L_true)

    # test cost function value
    mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
    tau = generate_textures_gamma_dist(N)
    sigma = generate_complex_covariance(p, unit_det=True)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    cost, _, _ = create_cost_egrad_ehess_nll_CG_constrained_scatter(X)

    L = cost(mu, sigma, tau)
    L_true = 0
    sigma_inv = la.inv(sigma)
    for i in range(N):
        x = X[:, i] - mu.reshape(-1)
        Q = np.real(x.conj().T@sigma_inv@x)
        L_true = L_true + p*np.log(tau[i])+Q/tau[i]
    L_true = np.real(L_true)
    assert_allclose(L, L_true)


def test_egrad_location_covariance_texture_constrained_scatter():
    rnd.seed(123)

    p = 3
    N = 20

    # test egrad
    mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
    tau = generate_textures_gamma_dist(N)
    sigma = generate_complex_covariance(p, unit_det=True)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    _, egrad, _ = create_cost_egrad_ehess_nll_CG_constrained_scatter(
        X, autodiff=False)
    _, egrad_num, _ = create_cost_egrad_ehess_nll_CG_constrained_scatter(
        X, autodiff=True)

    gc = egrad(mu, sigma, tau)
    gn = egrad_num(mu, sigma, tau)

    # test grad mu
    assert_allclose(gc[0], gn[0])
    # test grad sigma
    assert_allclose(gc[1], gn[1])
    # test grad tau
    assert_allclose(gc[2], gn[2])


def test_ehess_location_covariance_texture_constrained_scatter():
    rnd.seed(123)

    p = 3
    N = 20

    # test ehess
    mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
    tau = generate_textures_gamma_dist(N)
    sigma = generate_complex_covariance(p, unit_det=True)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    _, _, ehess = create_cost_egrad_ehess_nll_CG_constrained_scatter(
        X,
        autodiff=False
    )
    _, _, ehess_num = create_cost_egrad_ehess_nll_CG_constrained_scatter(
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
    assert_allclose(hc[0], hn[0])
    # test ehess sigma
    assert_allclose(hc[1], hn[1])
    # test ehess tau
    assert_allclose(hc[2], hn[2])


def test_real_location_covariance_texture_constrained_scatter():
    rnd.seed(123)

    N = int(1e3)
    p = 10
    feature = location_covariance_texture_constrained_scatter(
        iter_max=1000,
        solver='conjugate',
        information_geometry=False
    )(p, N)

    mu = rnd.normal(size=(p, 1))
    sigma = generate_covariance(p, unit_det=True)
    tau = generate_textures_gamma_dist(N, min_value=1e-16)
    X = sample_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.float64

    res = feature.estimation(X).export()
    assert res[0].dtype == np.float64
    assert res[1].dtype == np.float64
    assert res[2].dtype == np.float64
    assert la.norm(mu - res[0])/la.norm(mu) < 0.05
    # assert la.norm(sigma - res[1])/la.norm(sigma) < 0.4


def test_real_location_covariance_texture_IG_constrained_scatter():
    rnd.seed(123)

    N = int(1e3)
    p = 10
    feature = location_covariance_texture_constrained_scatter(
        iter_max=100,
        solver='conjugate',
        information_geometry=True
    )(p, N)

    mu = rnd.normal(size=(p, 1))
    sigma = generate_covariance(p, unit_det=True)
    tau = generate_textures_gamma_dist(N, nu=0.1)
    X = sample_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.float64

    res = feature.estimation(X).export()
    assert res[0].dtype == np.float64
    assert res[1].dtype == np.float64
    assert res[2].dtype == np.float64
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.1


def test_complex_location_covariance_texture_IG_constrained_scatter():
    rnd.seed(123)

    N = int(1e3)
    p = 10
    feature = location_covariance_texture_constrained_scatter(
        iter_max=100,
        solver='conjugate',
        information_geometry=True
    )(p, N)

    mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
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
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.1

# TEST OF FUNCTIONS WITH A CONSTRAINT OF UNITARY PRODUCT
# ON TEXTURES


def test_real_location_covariance_texture_Gaussian_constrained_texture():
    rnd.seed(123)

    N = int(1e6)
    p = 5
    feature = location_covariance_texture_Gaussian_constrained_texture()(p, N)

    mu = rnd.normal(size=(p, 1))
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


def test_complex_location_covariance_texture_Gaussian_constrained_texture():
    rnd.seed(123)

    N = int(1e6)
    p = 5
    feature = location_covariance_texture_Gaussian_constrained_texture()(p, N)

    mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
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


def test_cost_location_covariance_texture_constrained_texture():
    rnd.seed(123)

    p = 3
    N = 20

    # test cost function value when mu=0, tau=1 and sigma=I
    mu = np.zeros((p, 1), dtype=np.complex128)
    tau = np.ones(N)
    sigma = np.eye(p, dtype=np.complex128)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    cost, _, _ = create_cost_egrad_ehess_nll_CG_constrained_texture(X)

    L = cost(mu, sigma, tau)
    L_true = np.tensordot(X, X.conj(), X.ndim)/(p*N)
    assert_allclose(L, L_true)

    # test cost function value when mu=0, tau=1 and sigma=2*I
    mu = np.zeros((p, 1), dtype=np.complex128)
    tau = np.ones(N)
    sigma = 2*np.eye(p, dtype=np.complex128)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    mu_estimated = (1/N)*np.sum(X, axis=1, keepdims=True)
    scm = (1/N)*(X - mu_estimated)@(X - mu_estimated).conj().T
    kappa = np.trace(scm)/p

    # L1
    cost, _, _ = create_cost_egrad_ehess_nll_CG_constrained_texture(
        X, reg_type='L1', reg_beta=1, reg_kappa='trace_SCM')
    L = cost(mu, sigma, tau)
    L_true = N*p*np.log(2) + 0.5*np.tensordot(X, X.conj(), X.ndim)
    L_true += N*p*np.abs(2**(-1) - kappa**(-1))
    L_true /= p*N
    assert_allclose(L, L_true)

    # L2
    cost, _, _ = create_cost_egrad_ehess_nll_CG_constrained_texture(
        X, reg_type='L2', reg_beta=1, reg_kappa='trace_SCM')
    L = cost(mu, sigma, tau)
    L_true = N*p*np.log(2) + 0.5*np.tensordot(X, X.conj(), X.ndim)
    L_true += N*p*((2**(-1) - kappa**(-1))**2)
    L_true /= p*N
    assert_allclose(L, L_true)

    # AF
    cost, _, _ = create_cost_egrad_ehess_nll_CG_constrained_texture(
        X, reg_type='AF', reg_beta=1, reg_kappa='trace_SCM')
    L = cost(mu, sigma, tau)
    L_true = N*p*np.log(2) + 0.5*np.tensordot(X, X.conj(), X.ndim)
    L_true += N*p*((np.log(2) - np.log(kappa))**2)
    L_true /= p*N
    assert_allclose(L, L_true)

    # BW
    cost, _, _ = create_cost_egrad_ehess_nll_CG_constrained_texture(
        X, reg_type='BW', reg_beta=1, reg_kappa='trace_SCM')
    L = cost(mu, sigma, tau)
    L_true = N*p*np.log(2) + 0.5*np.tensordot(X, X.conj(), X.ndim)
    L_true += N*p*((2**(-0.5) - kappa**(-1/2))**2)
    L_true /= p*N
    assert_allclose(L, L_true)

    # KL
    cost, _, _ = create_cost_egrad_ehess_nll_CG_constrained_texture(
        X, reg_type='KL', reg_beta=1, reg_kappa='trace_SCM')
    L = cost(mu, sigma, tau)
    L_true = N*p*np.log(2) + 0.5*np.tensordot(X, X.conj(), X.ndim)
    L_true += N*p*(kappa*0.5 + np.log(2))
    L_true /= p*N
    assert_allclose(L, L_true)

    # test cost function value
    mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
    tau = generate_textures_gamma_dist(N, unit_prod=True)
    sigma = generate_complex_covariance(p)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    cost, _, _ = create_cost_egrad_ehess_nll_CG_constrained_texture(X)

    L = cost(mu, sigma, tau)
    Q = 0
    sigma_inv = la.inv(sigma)
    for i in range(N):
        x = X[:, i] - mu.reshape(-1)
        Q += np.real(x.conj().T@sigma_inv@x) / tau[i]
    L_true = N*np.log(la.det(sigma)) + Q
    L_true /= p*N
    assert_allclose(L, L_true)


def test_real_location_covariance_texture_constrained_texture():
    rnd.seed(123)

    N = int(1e3)
    p = 5
    feature = location_covariance_texture_constrained_texture(
        iter_max=100,
        solver='conjugate',
        information_geometry=False,
        reg_beta=0
    )(p, N)

    mu = rnd.normal(size=(p, 1))
    sigma = generate_covariance(p)
    tau = generate_textures_gamma_dist(N, nu=0.1, unit_prod=True)
    X = sample_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.float64

    res = feature.estimation(X).export()
    assert res[0].dtype == np.float64
    assert res[1].dtype == np.float64
    assert res[2].dtype == np.float64
    assert_allclose(np.prod(res[2]), 1)
    assert la.norm(mu - res[0])/la.norm(mu) < 0.3
    # we check that scatter matrices are close
    # sigma = sigma / (la.det(sigma)**(1/p))
    # res[1] = res[1] / (la.det(res[1])**(1/p))
    # assert la.norm(sigma - res[1])/la.norm(sigma) < 0.2


def test_real_location_covariance_texture_IG_constrained_texture():
    rnd.seed(123)

    N = int(1e3)
    p = 10
    feature = location_covariance_texture_constrained_texture(
        iter_max=100,
        solver='conjugate',
        information_geometry=True,
        reg_beta=0
    )(p, N)

    mu = rnd.normal(size=(p, 1))
    sigma = generate_covariance(p)
    tau = generate_textures_gamma_dist(
        N, nu=0.1, unit_prod=True, min_value=1e-16)
    X = sample_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.float64

    res = feature.estimation(X).export()
    assert res[0].dtype == np.float64
    assert res[1].dtype == np.float64
    assert res[2].dtype == np.float64
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.2
    sigma = sigma / (la.det(sigma)**(1/p))
    res[1] = res[1] / (la.det(res[1])**(1/p))
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.1


def test_complex_location_covariance_texture_IG_constrained_texture():
    rnd.seed(123)

    N = int(1e3)
    p = 10
    feature = location_covariance_texture_constrained_texture(
        iter_max=100,
        solver='conjugate',
        information_geometry=True,
        reg_beta=0
    )(p, N)

    mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
    sigma = generate_complex_covariance(p)
    tau = generate_textures_gamma_dist(N, nu=0.1, unit_prod=True)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.complex128

    res = feature.estimation(X).export()
    assert res[0].dtype == np.complex128
    assert res[1].dtype == np.complex128
    assert res[2].dtype == np.float64
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.25
    sigma = sigma / (la.det(sigma)**(1/p))
    res[1] = res[1] / (la.det(res[1])**(1/p))
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.1


def test_real_location_covariance_texture_constrained_texture_triangle():
    rnd.seed(123)

    N = int(1e3)
    p = 10
    weights = (0.9, 0.1)
    feature = location_covariance_texture_constrained_texture_triangle(
        iter_max=100,
        solver='conjugate',
        information_geometry=True,
        reg_beta=0,
        weights=weights
    )(p, N)

    mu = rnd.normal(size=(p, 1))
    sigma = generate_covariance(p)
    tau = generate_textures_gamma_dist(N, nu=0.1, unit_prod=True)
    X = sample_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.float64

    # estimation
    res = feature.estimation(X).export()
    assert res[0].dtype == np.float64
    assert res[0].shape == (p, 1)
    assert res[1].dtype == np.float64
    assert res[1].shape == (p, p)
    assert res[2].dtype == np.float64
    assert res[2].shape == (N, 1)
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.2
    sigma = sigma / (la.det(sigma)**(1/p))
    res[1] = res[1] / (la.det(res[1])**(1/p))
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.1

    # distance
    N = int(1e2)
    p = 10
    theta_1 = _FeatureArray((p, 1), (p, p), (N, 1))
    theta_1.append([
        rnd.normal(size=(p, 1)),
        generate_covariance(p),
        generate_textures_gamma_dist(N, nu=0.1, unit_prod=True)
    ])
    theta_2 = _FeatureArray((p, 1), (p, p), (N, 1))
    theta_2.append([
        rnd.normal(size=(p, 1)),
        generate_covariance(p),
        generate_textures_gamma_dist(N, nu=0.1, unit_prod=True)
    ])
    M_Gaussian = ComplexGaussianIG(p)
    p1 = [theta_1.export()[0], theta_1.export()[1]]
    p2 = [theta_2.export()[0], theta_2.export()[1]]
    dist = weights[0] * (M_Gaussian.dist(p1, p2)**2)
    dist = dist + weights[1] * (la.norm(np.log(theta_2.export()[2])
                                        - np.log(theta_1.export()[2]))**2)
    dist = np.sqrt(dist)
    assert_allclose(feature.distance(theta_1, theta_2), dist)

    # mean
    N_mean = 100
    N = 50
    p = 10
    theta = _FeatureArray((p, 1), (p, p), (N, 1))
    for _ in range(N_mean):
        theta.append([
            rnd.normal(size=(p, 1)),
            generate_covariance(p),
            generate_textures_gamma_dist(N, nu=0.1, unit_prod=True)
        ])

    tmp = feature.mean(theta).export()
    mean_Gaussian = [tmp[0], tmp[1]]
    assert mean_Gaussian[0].dtype == np.float64
    assert mean_Gaussian[0].shape == (p, 1)
    assert mean_Gaussian[1].dtype == np.float64
    assert mean_Gaussian[1].shape == (p, p)
    mean_tau = tmp[2]
    assert mean_tau.dtype == np.float64
    assert mean_tau.shape == (N, 1)

    feature_Gaussian = location_covariance_orth_triangle()(p, N)
    theta_Gaussian = _FeatureArray((p, 1), (p, p))
    theta_Gaussian.append([
        theta.export()[0],
        theta.export()[1]
    ])
    desired_mean_Gaussian = feature_Gaussian.mean(theta_Gaussian).export()
    assert_allclose(mean_Gaussian[0], desired_mean_Gaussian[0], rtol=1e-3)
    assert_allclose(mean_Gaussian[1], desired_mean_Gaussian[1], rtol=1e-3)

    desired_mean_tau = np.prod(theta.export()[2], axis=0)**(1/N_mean)
    assert_allclose(mean_tau, desired_mean_tau)


def test_complex_location_covariance_texture_constrained_texture_div_alpha():
    rnd.seed(123)

    N = int(1e3)
    p = 10
    feature = location_covariance_texture_constrained_texture_div_alpha(
        iter_max=100,
        solver='conjugate',
        information_geometry=True,
        reg_beta=0,
        alpha=0.5
    )(p, N)

    mu = rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1))
    sigma = generate_complex_covariance(p)
    tau = generate_textures_gamma_dist(N, nu=0.1, unit_prod=True)
    X = sample_complex_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.complex128

    # estimation
    res = feature.estimation(X).export()
    assert res[0].dtype == np.complex128
    assert res[0].shape == (p, 1)
    assert res[1].dtype == np.complex128
    assert res[1].shape == (p, p)
    assert res[2].dtype == np.float64
    assert res[2].shape == (N, 1)
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.2
    sigma = sigma / (la.det(sigma)**(1/p))
    res[1] = res[1] / (la.det(res[1])**(1/p))
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.1

    # mean
    # we only test the convergence of the mean computation
    N_mean = 100
    N = 50
    p = 10
    alpha = 0.5
    feature = location_covariance_texture_constrained_texture_div_alpha(
        alpha=alpha)
    feature = feature(p, N)
    theta = _FeatureArray((p, 1), (p, p), (N, 1))
    for _ in range(N_mean):
        tmp = [
            rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1)),
            generate_complex_covariance(p),
            generate_textures_gamma_dist(N, nu=1, unit_prod=True)
        ]
        theta.append(tmp)

    mean = feature.mean(theta).export()
    assert mean[0].dtype == np.complex128
    assert mean[0].shape == (p, 1)
    assert mean[1].dtype == np.complex128
    assert mean[1].shape == (p, p)
    assert mean[2].dtype == np.float64
    assert mean[2].shape == (N, 1)

    M = ComplexCompoundGaussianIGConstrainedTexture(p, N, k=1, alpha=alpha)
    theta = theta.export()

    def _cost(mean):
        mean_batch = [
            np.tile(
                mean[i],
                reps=(len(theta[0]), *([1]*mean[i].ndim))
            )
            for i in range(len(mean))
        ]
        d_squared = M.dist(mean_batch, theta)**2
        d_squared = (1/(2*len(theta[0]))) * d_squared
        return d_squared

    def _cost_diff(*mean):
        return _cost(mean)

    def _grad(mean):
        argnum = list(range(len(mean)))
        egrad = autograd.grad(_cost_diff, argnum=argnum)(*mean)
        egrad = list(egrad)
        for i in range(len(egrad)):
            egrad[i] = np.conjugate(egrad[i])
        grad = M.egrad2rgrad(mean, egrad)
        grad = _ProductTangentVector(grad)
        return grad

    grad = _grad(mean)

    assert M.norm(mean, grad) < feature._min_grad_norm


def test_real_location_covariance_texture_constrained_texture_div_alpha():
    rnd.seed(123)

    N = int(1e3)
    p = 10
    alpha = 0.33
    feature = location_covariance_texture_constrained_texture_div_alpha(
        iter_max=100,
        solver='conjugate',
        information_geometry=True,
        div_alpha_real_case=True,
        reg_beta=0,
        alpha=alpha
    )(p, N)
    M = ComplexCompoundGaussianIGConstrainedTexture(p, N, k=1, alpha=alpha)

    mu = rnd.normal(size=(p, 1))
    sigma = generate_covariance(p)
    tau = generate_textures_gamma_dist(N, nu=0.1, unit_prod=True)
    X = sample_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.float64

    # estimation
    res = feature.estimation(X).export()
    assert res[0].dtype == np.float64
    assert res[0].shape == (p, 1)
    assert res[1].dtype == np.float64
    assert res[1].shape == (p, p)
    assert res[2].dtype == np.float64
    assert res[2].shape == (N, 1)
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.2
    sigma = sigma / (la.det(sigma)**(1/p))
    res[1] = res[1] / (la.det(res[1])**(1/p))
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.1

    # div
    tmp1 = _FeatureArray((p, 1), (p, p), (N, 1))
    tmp1.append([
        rnd.normal(size=(p, 1)),
        generate_covariance(p),
        np.exp(rnd.normal(size=(N, 1))),
    ])
    tmp2 = _FeatureArray((p, 1), (p, p), (N, 1))
    tmp2.append([
        rnd.normal(size=(p, 1)),
        generate_covariance(p),
        np.exp(rnd.normal(size=(N, 1))),
    ])
    d1 = feature.distance(tmp1, tmp2)
    d2 = M.div_alpha_real_case(tmp1.export(), tmp2.export())
    assert_allclose(d1, d2)

    # mean
    # we only test the convergence of the mean computation
    N_mean = 100
    N = 50
    p = 10
    feature = location_covariance_texture_constrained_texture_div_alpha(
        iter_max=100,
        solver='conjugate',
        information_geometry=True,
        div_alpha_real_case=True,
        reg_beta=0,
        alpha=alpha
    )(p, N)
    theta = _FeatureArray((p, 1), (p, p), (N, 1))
    for _ in range(N_mean):
        tmp = [
            rnd.normal(size=(p, 1)),
            generate_covariance(p),
            generate_textures_gamma_dist(N, nu=1, unit_prod=True)
        ]
        theta.append(tmp)

    mean = feature.mean(theta).export()
    assert mean[0].dtype == np.float64
    assert mean[0].shape == (p, 1)
    assert mean[1].dtype == np.float64
    assert mean[1].shape == (p, p)
    assert mean[2].dtype == np.float64
    assert mean[2].shape == (N, 1)

    M = ComplexCompoundGaussianIGConstrainedTexture(p, N, k=1, alpha=alpha)
    theta = theta.export()

    def _cost(mean):
        mean_batch = [
            np.tile(
                mean[i],
                reps=(len(theta[0]), *([1]*mean[i].ndim))
            )
            for i in range(len(mean))
        ]
        d_squared = M.div_alpha_real_case(mean_batch, theta)**2
        d_squared = (1/(2*len(theta[0]))) * d_squared
        return d_squared

    def _cost_diff(*mean):
        return _cost(mean)

    def _grad(mean):
        argnum = list(range(len(mean)))
        egrad = autograd.grad(_cost_diff, argnum=argnum)(*mean)
        egrad = list(egrad)
        grad = M.egrad2rgrad(mean, egrad)
        grad = _ProductTangentVector(grad)
        return grad

    grad = _grad(mean)

    assert M.norm(mean, grad) < feature._min_grad_norm


def test_real_location_covariance_texture_constrained_texture_div_alpha_sym():
    rnd.seed(123)

    N = int(1e3)
    p = 10
    alpha = 0.33
    feature = location_covariance_texture_constrained_texture_div_alpha(
        iter_max=100,
        solver='conjugate',
        information_geometry=True,
        div_alpha_real_case=True,
        symmetrize_div=True,
        reg_beta=0,
        alpha=alpha
    )(p, N)
    M = ComplexCompoundGaussianIGConstrainedTexture(p, N, k=1, alpha=alpha)

    mu = rnd.normal(size=(p, 1))
    sigma = generate_covariance(p)
    tau = generate_textures_gamma_dist(N, nu=0.1, unit_prod=True)
    X = sample_compound_distribution(tau, sigma)
    X = X + mu
    assert X.dtype == np.float64

    # estimation
    res = feature.estimation(X).export()
    assert res[0].dtype == np.float64
    assert res[0].shape == (p, 1)
    assert res[1].dtype == np.float64
    assert res[1].shape == (p, p)
    assert res[2].dtype == np.float64
    assert res[2].shape == (N, 1)
    assert la.norm(mu - res[0])/la.norm(mu) < 0.01
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.2
    sigma = sigma / (la.det(sigma)**(1/p))
    res[1] = res[1] / (la.det(res[1])**(1/p))
    assert la.norm(sigma - res[1])/la.norm(sigma) < 0.1

    # div
    tmp1 = _FeatureArray((p, 1), (p, p), (N, 1))
    tmp1.append([
        rnd.normal(size=(p, 1)),
        generate_covariance(p),
        np.exp(rnd.normal(size=(N, 1))),
    ])
    tmp2 = _FeatureArray((p, 1), (p, p), (N, 1))
    tmp2.append([
        rnd.normal(size=(p, 1)),
        generate_covariance(p),
        np.exp(rnd.normal(size=(N, 1))),
    ])
    d1 = feature.distance(tmp1, tmp2)
    d2 = M.div_alpha_sym_real_case(tmp1.export(), tmp2.export())
    assert_allclose(d1, d2)

    # mean
    # we only test the convergence of the mean computation
    N_mean = 100
    N = 50
    p = 10
    feature = location_covariance_texture_constrained_texture_div_alpha(
        iter_max=100,
        solver='conjugate',
        information_geometry=True,
        div_alpha_real_case=True,
        reg_beta=0,
        alpha=alpha
    )(p, N)
    theta = _FeatureArray((p, 1), (p, p), (N, 1))
    for _ in range(N_mean):
        tmp = [
            rnd.normal(size=(p, 1)),
            generate_covariance(p),
            generate_textures_gamma_dist(N, nu=1, unit_prod=True)
        ]
        theta.append(tmp)

    mean = feature.mean(theta).export()
    assert mean[0].dtype == np.float64
    assert mean[0].shape == (p, 1)
    assert mean[1].dtype == np.float64
    assert mean[1].shape == (p, p)
    assert mean[2].dtype == np.float64
    assert mean[2].shape == (N, 1)

    M = ComplexCompoundGaussianIGConstrainedTexture(p, N, k=1, alpha=alpha)
    theta = theta.export()

    def _cost(mean):
        mean_batch = [
            np.tile(
                mean[i],
                reps=(len(theta[0]), *([1]*mean[i].ndim))
            )
            for i in range(len(mean))
        ]
        d_squared = M.div_alpha_real_case(mean_batch, theta)**2
        d_squared = (1/(2*len(theta[0]))) * d_squared
        return d_squared

    def _cost_diff(*mean):
        return _cost(mean)

    def _grad(mean):
        argnum = list(range(len(mean)))
        egrad = autograd.grad(_cost_diff, argnum=argnum)(*mean)
        egrad = list(egrad)
        grad = M.egrad2rgrad(mean, egrad)
        grad = _ProductTangentVector(grad)
        return grad

    grad = _grad(mean)

    assert M.norm(mean, grad) < feature._min_grad_norm
