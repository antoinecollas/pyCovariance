import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd

from pyCovariance.features import\
        center_euclidean
from pyCovariance.generation_data import\
        generate_covariance,\
        sample_normal_distribution
from pyCovariance.tangent_space import\
        TangentSpace
from pyCovariance.testing import assert_allclose


def helper_TangentSpace(ts):
    n_samples = int(1e3)
    p = 5
    N = 45

    X = np.zeros((n_samples, p, N))
    cov = generate_covariance(p)
    mu = rnd.randn(p, 1)
    for i in range(n_samples):
        X[i] = sample_normal_distribution(N, cov) + mu

    # fit
    ts.fit(X)
    mean = ts._mean.export()
    temp = X[:, :, N//2]
    mean_desired = np.mean(temp, axis=0)
    condition = la.norm(mean - mean_desired) / la.norm(mean_desired)
    assert condition < 0.01

    # transform
    X_ts = ts.transform(X)
    assert type(X_ts) is np.ndarray
    assert X_ts.dtype == np.float64
    assert X_ts.shape == (n_samples, p)
    temp = X[:, :, N//2]
    X_ts_desired = temp - mean_desired
    assert_allclose(X_ts, X_ts_desired)

    # fit_transform
    X_ts = ts.fit_transform(X)
    assert type(X_ts) is np.ndarray
    assert X_ts.dtype == np.float64
    assert X_ts.shape == (n_samples, p)
    assert_allclose(X_ts, X_ts_desired)

    # transform other data with same mean
    mu = 10*rnd.randn(p, 1)
    for i in range(n_samples):
        X[i] = sample_normal_distribution(N, cov) + mu
    X_ts = ts.transform(X)
    assert type(X_ts) is np.ndarray
    assert X_ts.dtype == np.float64
    assert X_ts.shape == (n_samples, p)
    temp = X[:, :, N//2]
    X_ts_desired = temp - mean_desired
    assert_allclose(X_ts, X_ts_desired)


def test_TangentSpace_single_job():
    rnd.seed(123)

    feature = center_euclidean()
    ts = TangentSpace(feature, n_jobs=1)
    helper_TangentSpace(ts)


def test_TangentSpace_multiple_job():
    rnd.seed(123)

    feature = center_euclidean()
    ts = TangentSpace(feature, n_jobs=-1)
    helper_TangentSpace(ts)
