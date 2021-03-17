import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd
import numpy.testing as np_test
import os

from pyCovariance.features import\
        center_euclidean
from pyCovariance.features.base import _FeatureArray
from pyCovariance.utils import\
        _compute_means,\
        _compute_pairwise_distances,\
        _estimate_features


def test__estimate_features():
    rnd.seed(123)

    p = 5
    N = int(1e2)
    n_samples = int(1e1)

    X = rnd.randn(n_samples, p, N)
    feature = center_euclidean()(p, N)

    # single thread
    f = _estimate_features(X,
                           feature.estimation,
                           n_jobs=1).export()
    assert f.dtype == np.float64
    assert f.shape == (n_samples, p)
    for i in range(n_samples):
        np_test.assert_almost_equal(f[i], X[i, :, N//2])

    # multiple threads
    f = _estimate_features(X,
                           feature.estimation,
                           n_jobs=-1).export()
    assert f.dtype == np.float64
    assert f.shape == (n_samples, p)
    for i in range(n_samples):
        np_test.assert_almost_equal(f[i], X[i, :, N//2])


def test__compute_means():
    rnd.seed(123)

    p = 5
    N = int(1e2)
    K = int(1e1)

    X = _FeatureArray((p, ))
    X.append(rnd.randn(N, p))
    C = rnd.randint(K, size=N)
    assert C.shape == (N, )
    feature = center_euclidean()(p, N)

    # single thread
    m = _compute_means(X,
                       C,
                       feature.mean,
                       n_jobs=1).export()
    assert m.dtype == np.float64
    assert m.shape == (K, p)
    for k in range(K):
        np_test.assert_almost_equal(m[k], np.mean(X[C == k].export(), axis=0))

    # single thread with init
    init = _FeatureArray((p, ))
    for _ in range(K):
        init.append(rnd.randn(p))
    m = _compute_means(X,
                       C,
                       feature.mean,
                       init=init,
                       n_jobs=1).export()
    assert m.dtype == np.float64
    assert m.shape == (K, p)
    for k in range(K):
        np_test.assert_almost_equal(m[k], np.mean(X[C == k].export(), axis=0))

    # multiple threads
    m = _compute_means(X,
                       C,
                       feature.mean,
                       n_jobs=os.cpu_count()).export()
    assert m.dtype == np.float64
    assert m.shape == (K, p)
    for k in range(K):
        np_test.assert_almost_equal(m[k], np.mean(X[C == k].export(), axis=0))


def test__compute_pairwise_distances():
    rnd.seed(123)

    p = 5
    N = int(1e2)
    N_mean = int(1e1)

    X = _FeatureArray((p, ))
    X.append(rnd.randn(N, p))
    mu = _FeatureArray((p, ))
    mu.append(rnd.randn(N_mean, p))
    feature = center_euclidean()(p, N)

    # single thread
    d = _compute_pairwise_distances(X,
                                    mu,
                                    feature.distance,
                                    n_jobs=1)
    assert d.dtype == np.float64
    assert d.shape == (N, N_mean)
    for i in range(N):
        for k in range(N_mean):
            assert d[i, k] == la.norm(X[i].export()-mu[k].export())

    # multiple thread
    d = _compute_pairwise_distances(X,
                                    mu,
                                    feature.distance,
                                    n_jobs=os.cpu_count())
    assert d.dtype == np.float64
    assert d.shape == (N, N_mean)
    for i in range(N):
        for k in range(N_mean):
            assert d[i, k] == la.norm(X[i].export()-mu[k].export())
