import autograd.numpy as np
import autograd.numpy.linalg as la
import numpy.testing as np_test
import os

from pyCovariance.clustering_functions import \
        compute_means_parallel,\
        compute_pairwise_distances_parallel
from pyCovariance.features import pixel_euclidean
from pyCovariance.features.base import _FeatureArray


def test_compute_pairwise_distances_parallel():
    p = 5
    N = int(1e2)
    N_mean = int(1e1)

    X = _FeatureArray((p, ))
    X.append(np.random.randn(N, p))
    mu = _FeatureArray((p, ))
    mu.append(np.random.randn(N_mean, p))
    pix = pixel_euclidean(p)

    # single thread
    d = compute_pairwise_distances_parallel(X,
                                            mu,
                                            pix.distance,
                                            enable_multi=False)
    assert d.dtype == np.float64
    assert d.shape == (N, N_mean)
    for i in range(N):
        for k in range(N_mean):
            assert d[i, k] == la.norm(X[i].export()-mu[k].export())

    # multiple thread
    d = compute_pairwise_distances_parallel(X,
                                            mu,
                                            pix.distance,
                                            enable_multi=True,
                                            number_of_threads=os.cpu_count())
    assert d.dtype == np.float64
    assert d.shape == (N, N_mean)
    for i in range(N):
        for k in range(N_mean):
            assert d[i, k] == la.norm(X[i].export()-mu[k].export())


def test_compute_means_parallel():
    p = 5
    N = int(1e2)
    K = int(1e1)

    X = _FeatureArray((p, ))
    X.append(np.random.randn(N, p))
    C = np.random.randint(K, size=N)
    assert C.shape == (N, )
    pix = pixel_euclidean(p)

    # single thread
    m = compute_means_parallel(X,
                               K,
                               C,
                               pix.mean,
                               enable_multi=False).export()
    assert m.dtype == np.float64
    assert m.shape == (K, p)
    for k in range(K):
        np_test.assert_almost_equal(m[k], np.mean(X[C == k].export(), axis=0))

    # multiple threads
    m = compute_means_parallel(X,
                               K,
                               C,
                               pix.mean,
                               enable_multi=True).export()
    assert m.dtype == np.float64
    assert m.shape == (K, p)
    for k in range(K):
        np_test.assert_almost_equal(m[k], np.mean(X[C == k].export(), axis=0))
