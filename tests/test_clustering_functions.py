import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd
import numpy.testing as np_test
import os
import sys

from pyCovariance import K_means
from pyCovariance.clustering_functions import \
        compute_objective_function,\
        compute_means_parallel,\
        compute_pairwise_distances_parallel,\
        random_index_for_initialisation
from pyCovariance.features import pixel_euclidean
from pyCovariance.features.base import _FeatureArray
from pyCovariance.generation_data import \
        generate_covariance,\
        sample_normal_distribution


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
                                            nb_threads=os.cpu_count())
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
                               C,
                               pix.mean,
                               nb_threads=1).export()
    assert m.dtype == np.float64
    assert m.shape == (K, p)
    for k in range(K):
        np_test.assert_almost_equal(m[k], np.mean(X[C == k].export(), axis=0))

    # multiple threads
    m = compute_means_parallel(X,
                               C,
                               pix.mean,
                               nb_threads=os.cpu_count()).export()
    assert m.dtype == np.float64
    assert m.shape == (K, p)
    for k in range(K):
        np_test.assert_almost_equal(m[k], np.mean(X[C == k].export(), axis=0))


def test_random_index_for_initialisation():
    K = 10
    N = 20
    idx = random_index_for_initialisation(K, N)
    assert len(np.unique(idx)) == len(idx)


def test_compute_objective_function():
    N = int(1e3)
    K = 10
    distances = rnd.rand(N, K)
    res = 0
    for i in range(distances.shape[0]):
        d = distances[i]
        k = np.argmin(d)
        res += d[k]**2
    var_intr = compute_objective_function(distances)
    np_test.assert_almost_equal(res, var_intr)


def test_K_means():
    N = 50
    p = 2

    # generating data points to cluster
    X = _FeatureArray((p, ))
    cov1 = generate_covariance(p)
    temp = sample_normal_distribution(N, cov1) + 2*np.ones((p, 1))
    X.append(temp.T)
    cov2 = generate_covariance(p)
    temp = sample_normal_distribution(N, cov2) - 2*np.ones((p, 1))
    X.append(temp.T)
    y = np.concatenate([np.zeros(N), np.ones(N)])
    idx = np.random.permutation(np.arange(2*N))
    y = y[idx]
    X = X[idx]

    # scatter plot of X
    # import matplotlib.pyplot as plt
    # plt.scatter(X.export()[:, 0], X.export()[:, 1], c=y)
    # plt.show()

    pix = pixel_euclidean(p)

    # single thread
    y_predict = K_means(
        X,
        K=2,
        distance=pix.distance,
        mean_function=pix.mean,
        init=None,
        enable_multi_distance=False,
        enable_multi_mean=False,
        nb_threads=1,
        verbose=False
    )[0]
    precision = np.sum(y == y_predict)/(2*N)
    if precision < 0.5:
        y_predict = np.mod(y_predict+1, 2)
    precision = np.sum(y == y_predict)/(2*N)
    assert precision >= 0.95

    # single thread with init
    init = np.concatenate([np.zeros(N), np.ones(N)])
    y_predict = K_means(
        X,
        K=2,
        distance=pix.distance,
        mean_function=pix.mean,
        init=init,
        enable_multi_distance=False,
        enable_multi_mean=False,
        nb_threads=1,
        verbose=False
    )[0]
    precision = np.sum(y == y_predict)/(2*N)
    if precision < 0.5:
        y_predict = np.mod(y_predict+1, 2)
    precision = np.sum(y == y_predict)/(2*N)
    assert precision >= 0.95

    # single thread with verbose
    sys.stdout = open(os.devnull, 'w')
    y_predict = K_means(
        X,
        K=2,
        distance=pix.distance,
        mean_function=pix.mean,
        init=None,
        enable_multi_distance=False,
        enable_multi_mean=False,
        nb_threads=1,
        verbose=True
    )[0]
    sys.stdout = sys.__stdout__
    precision = np.sum(y == y_predict)/(2*N)
    if precision < 0.5:
        y_predict = np.mod(y_predict+1, 2)
    precision = np.sum(y == y_predict)/(2*N)
    assert precision >= 0.95

    # multiple threads
    y_predict = K_means(
        X,
        K=2,
        distance=pix.distance,
        mean_function=pix.mean,
        init=None,
        enable_multi_distance=True,
        enable_multi_mean=True,
        nb_threads=os.cpu_count(),
        verbose=False
    )[0]
    precision = np.sum(y == y_predict)/(2*N)
    if precision < 0.5:
        y_predict = np.mod(y_predict+1, 2)
    precision = np.sum(y == y_predict)/(2*N)
    assert precision >= 0.95
