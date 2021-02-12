import autograd.numpy as np
import autograd.numpy.linalg as la
import numpy.testing as np_test

from pyCovariance.features import covariance, covariance_texture
from pyCovariance.features.base import _FeatureArray
from pyCovariance.generation_data import\
        generate_complex_covariance,\
        generate_covariance,\
        generate_textures,\
        sample_complex_compound_distribution,\
        sample_compound_distribution
from pyCovariance.matrix_operators import invsqrtm, logm, sqrtm


def test_real_covariance_texture():
    p = 5
    N = int(2*1e5)
    feature = covariance_texture()(p, N)
    assert type(str(feature)) is str

    # test estimation
    tau = generate_textures(N)
    sigma = generate_covariance(p, unit_det=True)
    sigma = sigma/(la.det(sigma)**(1/p))
    X = sample_compound_distribution(tau, sigma)

    res = feature.estimation(X).export()
    assert res[0].dtype == np.float64
    assert res[0].shape == (p, p)
    assert res[1].dtype == np.float64
    assert res[1].shape == (N, 1)
    assert la.norm(res[0]-sigma)/la.norm(sigma) < 0.01
    np_test.assert_almost_equal(np.abs(la.det(res[0])), 1)

    # test distance
    data = _FeatureArray((p, p), (N, 1))
    data.append([generate_covariance(p, unit_det=True),
                 generate_textures(N)])
    data.append([generate_covariance(p, unit_det=True),
                 generate_textures(N)])

    sigma0_isqrtm = invsqrtm(data[0].export()[0])
    prod = sigma0_isqrtm@data[1].export()[0]@sigma0_isqrtm
    eigvals = la.eigvalsh(prod)
    d = (1/p)*np.sum(np.log(eigvals)**2)
    temp = np.log(data[0].export()[1]) - np.log(data[1].export()[1])
    d += (1/N) * la.norm(temp)**2
    d = np.sqrt(d)
    np_test.assert_almost_equal(feature.distance(data[0], data[1]), d)

    # test distances
    data = _FeatureArray((p, p), (N, 1))
    data.append([generate_covariance(p, unit_det=True),
                 generate_textures(N)])
    data.append([generate_covariance(p, unit_det=True),
                 generate_textures(N)])

    sigma0_isqrtm = invsqrtm(data[0].export()[0])
    prod = sigma0_isqrtm@data[1].export()[0]@sigma0_isqrtm
    eigvals = la.eigvalsh(prod)
    d_sigma = np.sqrt(np.sum(np.log(eigvals)**2))

    temp = np.log(data[0].export()[1]) - np.log(data[1].export()[1])
    d_tau = la.norm(temp)

    d_M = np.sqrt((1/p) * d_sigma**2 + (1/N) * d_tau**2)
    d = [d_M, d_sigma, d_tau]

    np_test.assert_almost_equal(feature.distances(data[0], data[1]), d)

    # test log
    X = _FeatureArray((p, p), (N, 1))
    X.append([generate_covariance(p, unit_det=True),
              generate_textures(N)])
    Y = _FeatureArray((p, p), (N, 1))
    Y.append([generate_covariance(p, unit_det=True),
              generate_textures(N)])

    log = feature.log(X, Y)
    assert type(log) is _FeatureArray
    assert len(log) == 1
    log = log.export()
    assert type(log) is list
    assert len(log) == 2
    assert log[0].dtype == np.float64
    assert log[0].shape == (p, p)
    assert log[1].dtype == np.float64
    assert log[1].shape == (N, 1)
    X_sqrtm = sqrtm(X.export()[0])
    X_isqrtm = invsqrtm(X.export()[0])
    temp = X_isqrtm @ Y.export()[0] @ X_isqrtm
    temp = logm(temp)
    desired_log = [X_sqrtm @ temp @ X_sqrtm]
    temp = X.export()[1] * np.log(1/X.export()[1] * Y.export()[1])
    desired_log.append(temp)
    condition = la.norm(log[0]-desired_log[0])
    condition = condition / la.norm(desired_log[0])
    assert condition < 1e-3
    condition = la.norm(log[1]-desired_log[1])
    condition = condition / la.norm(desired_log[1])
    assert condition < 1e-3

    # test vectorized log
    log = feature.log(X, Y, vectorize=True)
    assert type(log) is np.ndarray
    assert log.dtype == np.float64
    assert log.shape == (1, p*(p+1)/2+N)
    desired_log[0] = desired_log[0][np.triu_indices(p)][np.newaxis, ...]
    desired_log[1] = desired_log[1].reshape((1, -1))
    desired_log = np.concatenate(desired_log, axis=1)
    np_test.assert_almost_equal(log, desired_log)

    # test batch log and vectorized log
    batch_size = 10
    X = _FeatureArray((p, p), (N, 1))
    X.append([generate_covariance(p, unit_det=True),
              generate_textures(N)])
    Y = _FeatureArray((p, p), (N, 1))
    for _ in range(batch_size):
        Y.append([generate_covariance(p, unit_det=True),
                  generate_textures(N)])

    log = feature.log(X, Y)
    assert type(log) is _FeatureArray
    assert len(log) == batch_size
    log = log.export()
    assert type(log) is list
    assert len(log) == 2
    assert log[0].dtype == np.float64
    assert log[0].shape == (batch_size, p, p)
    assert log[1].dtype == np.float64
    assert log[1].shape == (batch_size, N, 1)
    log_vec = feature.log(X, Y, vectorize=True)
    assert type(log_vec) is np.ndarray
    assert log_vec.dtype == np.float64
    assert log_vec.shape == (batch_size, p*(p+1)/2+N)
    for i in range(batch_size):
        X_sqrtm = sqrtm(X.export()[0])
        X_isqrtm = invsqrtm(X.export()[0])
        temp = X_isqrtm @ Y.export()[0][i] @ X_isqrtm
        temp = logm(temp)
        desired_log = [X_sqrtm @ temp @ X_sqrtm]
        temp = X.export()[1] * np.log(1/X.export()[1]*Y.export()[1][i])
        desired_log.append(temp)
        condition = la.norm(log[0][i]-desired_log[0])
        condition = condition / la.norm(desired_log[0])
        assert condition < 1e-3
        condition = la.norm(log[1][i]-desired_log[1])
        condition = condition / la.norm(desired_log[1])
        assert condition < 1e-3

        desired_log[0] = desired_log[0][np.triu_indices(p)][np.newaxis, ...]
        desired_log[1] = desired_log[1].reshape((1, -1))
        desired_log = np.concatenate(desired_log, axis=1)
        np_test.assert_almost_equal(log_vec[i], desired_log.reshape(-1))

    # test mean
    N = 5
    N_mean = 10
    data = _FeatureArray((p, p), (N, 1))
    for _ in range(N_mean):
        data.append([generate_covariance(p, unit_det=True),
                     generate_textures(N)])

    cov = covariance()(p, N)
    sigma = _FeatureArray((p, p))
    sigma.append(data.export()[0])
    mean_sigma = cov.mean(sigma).export()
    mean_text = np.prod(data.export()[1], axis=0)**(1/N_mean)
    m = feature.mean(data).export()
    np_test.assert_almost_equal(la.det(m[0]), 1)
    assert la.norm(m[0] - mean_sigma) / la.norm(mean_sigma) < 1e-6
    assert la.norm(m[0] - mean_sigma) / la.norm(mean_sigma) < 1e-6
    assert la.norm(m[1] - mean_text) / la.norm(mean_text) < 1e-6


def test_complex_covariance_texture():
    p = 5
    N = int(1e5)
    feature = covariance_texture()(p, N)
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

    # test distances
    data = _FeatureArray((p, p), (N, 1))
    data.append([generate_complex_covariance(p, unit_det=True),
                 generate_textures(N)])
    data.append([generate_complex_covariance(p, unit_det=True),
                 generate_textures(N)])

    sigma0_isqrtm = invsqrtm(data[0].export()[0])
    prod = sigma0_isqrtm@data[1].export()[0]@sigma0_isqrtm
    eigvals = la.eigvalsh(prod)
    d_sigma = np.sqrt(np.sum(np.log(eigvals)**2))

    temp = np.log(data[0].export()[1]) - np.log(data[1].export()[1])
    d_tau = la.norm(temp)

    d_M = np.sqrt((1/p) * d_sigma**2 + (1/N) * d_tau**2)
    d = [d_M, d_sigma, d_tau]

    np_test.assert_almost_equal(feature.distances(data[0], data[1]), d)

    # test log
    X = _FeatureArray((p, p), (N, 1))
    X.append([generate_complex_covariance(p, unit_det=True),
              generate_textures(N)])
    Y = _FeatureArray((p, p), (N, 1))
    Y.append([generate_complex_covariance(p, unit_det=True),
              generate_textures(N)])

    log = feature.log(X, Y)
    assert type(log) is _FeatureArray
    assert len(log) == 1
    log = log.export()
    assert type(log) is list
    assert len(log) == 2
    assert log[0].dtype == np.complex128
    assert log[0].shape == (p, p)
    assert log[1].dtype == np.float64
    assert log[1].shape == (N, 1)
    X_sqrtm = sqrtm(X.export()[0])
    X_isqrtm = invsqrtm(X.export()[0])
    temp = X_isqrtm @ Y.export()[0] @ X_isqrtm
    temp = logm(temp)
    desired_log = [X_sqrtm @ temp @ X_sqrtm]
    temp = X.export()[1] * np.log(1/X.export()[1] * Y.export()[1])
    desired_log.append(temp)
    condition = la.norm(log[0]-desired_log[0])
    condition = condition / la.norm(desired_log[0])
    assert condition < 1e-3
    condition = la.norm(log[1]-desired_log[1])
    condition = condition / la.norm(desired_log[1])
    assert condition < 1e-3

    # test vectorized log
    log = feature.log(X, Y, vectorize=True)
    assert type(log) is np.ndarray
    assert log.dtype == np.complex128
    assert log.shape == (1, p*(p+1)/2+N)
    desired_log[0] = desired_log[0][np.triu_indices(p)][np.newaxis, ...]
    desired_log[1] = desired_log[1].reshape((1, -1))
    desired_log = np.concatenate(desired_log, axis=1)
    np_test.assert_almost_equal(log, desired_log)

    # test batch log and vectorized log
    batch_size = 10
    X = _FeatureArray((p, p), (N, 1))
    X.append([generate_complex_covariance(p, unit_det=True),
              generate_textures(N)])
    Y = _FeatureArray((p, p), (N, 1))
    for _ in range(batch_size):
        Y.append([generate_complex_covariance(p, unit_det=True),
                  generate_textures(N)])

    log = feature.log(X, Y)
    assert type(log) is _FeatureArray
    assert len(log) == batch_size
    log = log.export()
    assert type(log) is list
    assert len(log) == 2
    assert log[0].dtype == np.complex128
    assert log[0].shape == (batch_size, p, p)
    assert log[1].dtype == np.float64
    assert log[1].shape == (batch_size, N, 1)
    log_vec = feature.log(X, Y, vectorize=True)
    assert type(log_vec) is np.ndarray
    assert log_vec.dtype == np.complex128
    assert log_vec.shape == (batch_size, p*(p+1)/2+N)
    for i in range(batch_size):
        X_sqrtm = sqrtm(X.export()[0])
        X_isqrtm = invsqrtm(X.export()[0])
        temp = X_isqrtm @ Y.export()[0][i] @ X_isqrtm
        temp = logm(temp)
        desired_log = [X_sqrtm @ temp @ X_sqrtm]
        temp = X.export()[1] * np.log(1/X.export()[1]*Y.export()[1][i])
        desired_log.append(temp)
        condition = la.norm(log[0][i]-desired_log[0])
        condition = condition / la.norm(desired_log[0])
        assert condition < 1e-3
        condition = la.norm(log[1][i]-desired_log[1])
        condition = condition / la.norm(desired_log[1])
        assert condition < 1e-3

        desired_log[0] = desired_log[0][np.triu_indices(p)][np.newaxis, ...]
        desired_log[1] = desired_log[1].reshape((1, -1))
        desired_log = np.concatenate(desired_log, axis=1)
        np_test.assert_almost_equal(log_vec[i], desired_log.reshape(-1))

    # test mean
    N = 5
    N_mean = 10
    data = _FeatureArray((p, p), (N, 1))
    for _ in range(N_mean):
        data.append([generate_complex_covariance(p, unit_det=True),
                     generate_textures(N)])

    cov = covariance()(p, N)
    sigma = _FeatureArray((p, p))
    sigma.append(data.export()[0])
    mean_sigma = cov.mean(sigma).export()
    mean_text = np.prod(data.export()[1], axis=0)**(1/N_mean)

    m = feature.mean(data).export()
    np_test.assert_almost_equal(la.det(m[0]), 1)
    assert la.norm(m[0] - mean_sigma) / la.norm(mean_sigma) < 1e-6
    assert la.norm(m[0] - mean_sigma) / la.norm(mean_sigma) < 1e-6
    assert la.norm(m[1] - mean_text) / la.norm(mean_text) < 1e-6
