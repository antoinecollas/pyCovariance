import autograd.numpy as np
import autograd.numpy.linalg as la
import numpy.testing as np_test

from pyCovariance.features import covariance, covariance_texture
from pyCovariance.features.base import _FeatureArray
from pyCovariance.generation_data import\
        generate_complex_covariance,\
        generate_textures,\
        sample_complex_compound_distribution
from pyCovariance.matrix_operators import invsqrtm


def test_complex_covariance_texture():
    p = 5
    N = int(1e6)
    feature = covariance_texture(p, N)
    assert type(str(feature)) is str

    # test estimation
    tau = generate_textures(N)
    sigma = generate_complex_covariance(p, unit_det=True)
    sigma = sigma/(la.det(sigma)**(1/p))
    X = sample_complex_compound_distribution(tau, sigma)

    res = feature.estimation(X).export()
    assert res[0].dtype == np.complex128
    assert res[0].shape == (p, p)
    assert res[1].dtype == np.float64
    assert res[1].shape == (N, 1)
    assert la.norm(res[0]-sigma)/la.norm(sigma) < 0.01
    np_test.assert_almost_equal(np.abs(la.det(res[0])), 1)

    # test distance
    data = _FeatureArray((p, p), (N, 1))
    data.append([generate_complex_covariance(p, unit_det=True),
                 generate_textures(N)])
    data.append([generate_complex_covariance(p, unit_det=True),
                 generate_textures(N)])

    sigma0_isqrtm = invsqrtm(data[0].export()[0])
    prod = sigma0_isqrtm@data[1].export()[0]@sigma0_isqrtm
    eigvals = la.eigvalsh(prod)
    d = (1/p)*np.sum(np.log(eigvals)**2)
    temp = np.log(data[0].export()[1]) - np.log(data[1].export()[1])
    d += (1/N) * la.norm(temp)**2
    d = np.sqrt(d)
    np_test.assert_almost_equal(feature.distance(data[0], data[1]), d)

    # test mean
    N = 5
    N_mean = 10
    data = _FeatureArray((p, p), (N, 1))
    for _ in range(N_mean):
        data.append([generate_complex_covariance(p, unit_det=True),
                     generate_textures(N)])

    cov = covariance(p)
    sigma = _FeatureArray((p, p))
    sigma.append(data.export()[0])
    mean_sigma = cov.mean(sigma).export()
    mean_text = np.prod(data.export()[1], axis=0)**(1/N_mean)

    m = feature.mean(data).export()
    np_test.assert_almost_equal(la.det(m[0]), 1)
    assert la.norm(m[0] - mean_sigma) / la.norm(mean_sigma) < 1e-4
    assert la.norm(m[1] - mean_text) / la.norm(mean_text) < 1e-4
