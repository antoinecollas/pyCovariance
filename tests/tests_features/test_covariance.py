import numpy as np
import numpy.linalg as la
import numpy.testing as np_test

from pyCovariance.matrix_operators import invsqrtm, logm, sqrtm
from pyCovariance.features.base import _FeatureArray
from pyCovariance.features import covariance
from pyCovariance.generation_data import generate_complex_covariance,\
        generate_covariance,\
        sample_complex_normal_distribution,\
        sample_complex_standard_normal_distribution,\
        sample_normal_distribution,\
        sample_standard_normal_distribution


def test_real_covariance():
    p = 5
    N = int(1e6)
    N_mean = 10
    cov = covariance(p)
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
    np_test.assert_almost_equal(cov.distance(sigma[0], sigma[1]), d)

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
    assert la.norm(m-m_closed_form)/la.norm(m_closed_form) < 1e-10

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
    assert condition < 1e-8


def test_complex_covariance():
    p = 5
    N = int(1e6)
    N_mean = 10
    cov = covariance(p)
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
    np_test.assert_almost_equal(cov.distance(sigma[0], sigma[1]), d)

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
    assert la.norm(m-m_closed_form)/la.norm(m_closed_form) < 1e-10

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
    assert condition < 1e-8
