import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd
import numpy.testing as np_test
import os

from pyCovariance import K_means

from pyCovariance.clustering_functions import \
        compute_objective_function,\
        compute_means_parallel,\
        compute_pairwise_distances_parallel,\
        _K_means,\
        random_index_for_initialisation

from pyCovariance.features import\
        center_euclidean,\
        covariance

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
    pix = center_euclidean(p)

    # single thread
    d = compute_pairwise_distances_parallel(X,
                                            mu,
                                            pix.distance,
                                            nb_threads=0)
    assert d.dtype == np.float64
    assert d.shape == (N, N_mean)
    for i in range(N):
        for k in range(N_mean):
            assert d[i, k] == la.norm(X[i].export()-mu[k].export())

    # multiple thread
    d = compute_pairwise_distances_parallel(X,
                                            mu,
                                            pix.distance,
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
    pix = center_euclidean(p)

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


def test__K_means():
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

    pix = center_euclidean(p)

    # single thread
    y_predict = _K_means(
        X,
        K=2,
        distance=pix.distance,
        mean_function=pix.mean,
        init=None,
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
    y_predict = _K_means(
        X,
        K=2,
        distance=pix.distance,
        mean_function=pix.mean,
        init=init,
        nb_threads=1,
        verbose=False
    )[0]
    precision = np.sum(y == y_predict)/(2*N)
    if precision < 0.5:
        y_predict = np.mod(y_predict+1, 2)
    precision = np.sum(y == y_predict)/(2*N)
    assert precision >= 0.95

    # multiple threads
    y_predict = _K_means(
        X,
        K=2,
        distance=pix.distance,
        mean_function=pix.mean,
        init=None,
        nb_threads=os.cpu_count(),
        verbose=False
    )[0]
    precision = np.sum(y == y_predict)/(2*N)
    if precision < 0.5:
        y_predict = np.mod(y_predict+1, 2)
    precision = np.sum(y == y_predict)/(2*N)
    assert precision >= 0.95

    # test final criteria and mu
    # generating data points to cluster
    X = _FeatureArray((p, ))
    cov1 = generate_covariance(p)
    temp = sample_normal_distribution(N, cov1)
    X.append(temp.T)
    cov2 = generate_covariance(p)
    temp = sample_normal_distribution(N, cov2)
    X.append(temp.T)
    y = np.concatenate([np.zeros(N), np.ones(N)])
    idx = np.random.permutation(np.arange(2*N))
    y = y[idx]
    X = X[idx]
    y_predict, mu, _, _, criterion_values = _K_means(
        X,
        K=2,
        distance=pix.distance,
        mean_function=pix.mean,
        init=None,
        nb_init=20,
        nb_threads=1,
        iter_max=1,
        verbose=False
    )
    best_criterion_value = np.inf
    for i in range(len(criterion_values)):
        for j in range(len(criterion_values[i])):
            if best_criterion_value > criterion_values[i][j]:
                best_criterion_value = criterion_values[i][j]
    d = compute_pairwise_distances_parallel(
        X,
        mu,
        pix.distance,
        nb_threads=1
    )
    criterion_value = compute_objective_function(d)
    assert np.abs(best_criterion_value - criterion_value) < 1e-8


def test_K_means():
    batch_size = 100
    N = 200
    p = 3

    X = np.zeros((batch_size, p, N))

    # generating points of class 1
    cov1 = generate_covariance(p)
    for i in range(batch_size//2):
        X[i] = sample_normal_distribution(N, cov1)

    # generating points of class 2
    cov2 = generate_covariance(p)
    for i in range(batch_size//2, batch_size):
        X[i] = sample_normal_distribution(N, cov2)

    # labels
    y = np.concatenate([
        np.zeros(batch_size//2),
        np.ones(batch_size - batch_size//2)
    ]).astype(int)

    assert X.shape[0] == len(y)

    # generating data points to cluster
    idx = np.random.permutation(batch_size)
    X = X[idx]
    y = y[idx]

    feature = covariance(p)

    # single thread
    y_predict = K_means(
        X,
        K=2,
        feature=feature,
        init=None,
        nb_init=10,
        iter_max=100,
        nb_threads=1,
        verbose=False
    )[0]
    precision = np.sum(y == y_predict)/batch_size
    if precision < 0.5:
        y_predict = np.mod(y_predict+1, 2)
    precision = np.sum(y == y_predict)/batch_size
    assert precision >= 0.8

    # single thread with init
    init = np.concatenate([
        np.zeros(batch_size//2),
        np.ones(batch_size - batch_size//2)
    ]).astype(int)
    y_predict = K_means(
        X,
        K=2,
        feature=feature,
        init=init,
        nb_threads=1,
        verbose=False
    )[0]
    precision = np.sum(y == y_predict)/batch_size
    if precision < 0.5:
        y_predict = np.mod(y_predict+1, 2)
    precision = np.sum(y == y_predict)/batch_size
    assert precision >= 0.8

    # multiple threads
    y_predict = K_means(
        X,
        K=2,
        feature=feature,
        init=None,
        nb_threads=os.cpu_count(),
        verbose=False
    )[0]
    precision = np.sum(y == y_predict)/batch_size
    if precision < 0.5:
        y_predict = np.mod(y_predict+1, 2)
    precision = np.sum(y == y_predict)/batch_size
    assert precision >= 0.8
