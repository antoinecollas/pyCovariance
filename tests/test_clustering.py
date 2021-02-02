import autograd.numpy as np
import autograd.numpy.random as rnd
import numpy.testing as np_test
import os

from pyCovariance import K_means

from pyCovariance.clustering import \
        _compute_objective_function,\
        _compute_pairwise_distances,\
        _K_means,\
        _random_index_for_initialisation

from pyCovariance.features import\
        center_euclidean,\
        covariance

from pyCovariance.features.base import _FeatureArray

from pyCovariance.generation_data import \
        generate_covariance,\
        sample_normal_distribution


def test__compute_objective_function():
    N = int(1e3)
    K = 10
    distances = rnd.rand(N, K)
    res = 0
    for i in range(distances.shape[0]):
        d = distances[i]
        k = np.argmin(d)
        res += d[k]**2
    var_intr = _compute_objective_function(distances)
    np_test.assert_almost_equal(res, var_intr)


def test__random_index_for_initialisation():
    K = 10
    N = 20
    idx = _random_index_for_initialisation(K, N)
    assert len(np.unique(idx)) == len(idx)


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

    pix = center_euclidean()(p, N)

    # single thread
    y_pred = _K_means(
        X,
        n_clusters=2,
        distance=pix.distance,
        mean_function=pix.mean,
        init=None,
        n_jobs=1,
        verbose=False
    )[0]
    precision = np.sum(y == y_pred)/(2*N)
    if precision < 0.5:
        y_pred = np.mod(y_pred+1, 2)
    precision = np.sum(y == y_pred)/(2*N)
    assert precision >= 0.95

    # single thread with init
    init = np.concatenate([np.zeros(N), np.ones(N)])
    y_pred = _K_means(
        X,
        n_clusters=2,
        distance=pix.distance,
        mean_function=pix.mean,
        init=init,
        n_jobs=1,
        verbose=False
    )[0]
    precision = np.sum(y == y_pred)/(2*N)
    if precision < 0.5:
        y_pred = np.mod(y_pred+1, 2)
    precision = np.sum(y == y_pred)/(2*N)
    assert precision >= 0.95

    # multiple threads
    y_pred = _K_means(
        X,
        n_clusters=2,
        distance=pix.distance,
        mean_function=pix.mean,
        init=None,
        n_jobs=os.cpu_count(),
        verbose=False
    )[0]
    precision = np.sum(y == y_pred)/(2*N)
    if precision < 0.5:
        y_pred = np.mod(y_pred+1, 2)
    precision = np.sum(y == y_pred)/(2*N)
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
    y_pred, mu, _, _, criterion_values = _K_means(
        X,
        n_clusters=2,
        distance=pix.distance,
        mean_function=pix.mean,
        init=None,
        n_init=20,
        n_jobs=1,
        max_iter=1,
        verbose=False
    )
    best_criterion_value = np.inf
    for i in range(len(criterion_values)):
        for j in range(len(criterion_values[i])):
            if best_criterion_value > criterion_values[i][j]:
                best_criterion_value = criterion_values[i][j]
    d = _compute_pairwise_distances(
        X,
        mu,
        pix.distance,
        n_jobs=1
    )
    criterion_value = _compute_objective_function(d)
    assert np.abs(best_criterion_value - criterion_value) < 1e-8


def test_K_means():
    n_samples = 100
    N = 200
    p = 3

    X = np.zeros((n_samples, p, N))

    # generating points of class 1
    cov1 = generate_covariance(p)
    for i in range(n_samples//2):
        X[i] = sample_normal_distribution(N, cov1)

    # generating points of class 2
    cov2 = generate_covariance(p)
    for i in range(n_samples//2, n_samples):
        X[i] = sample_normal_distribution(N, cov2)

    # labels
    y = np.concatenate([
        np.zeros(n_samples//2),
        np.ones(n_samples - n_samples//2)
    ]).astype(int)

    assert X.shape[0] == len(y)

    # generating data points to cluster
    idx = np.random.permutation(n_samples)
    X = X[idx]
    y = y[idx]

    feature = covariance()

    # single thread
    model = K_means(
        feature,
        n_clusters=2,
        max_iter=100,
        n_init=10,
        n_jobs=1,
        tol=1e-4,
        verbose=False
    )
    # fit
    model.fit(X)
    # predict
    y_pred = model.predict(X)
    precision = np.sum(y == y_pred)/n_samples
    if precision < 0.5:
        y_pred = np.mod(y_pred+1, 2)
    precision = np.sum(y == y_pred)/n_samples
    assert precision >= 0.9
    # transform
    y_pred = model.predict(X)
    d = model.transform(X)
    assert d.shape == (len(X), 2)
    y_pred_2 = d.argmin(axis=1)
    assert (y_pred_2 == y_pred).all()
    # predict_proba
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert (proba >= 0).all()
    actual = np.sum(proba, axis=1)
    desired = np.ones(len(X))
    assert ((actual - desired) < 1e-8).all()
    actual = proba.argmax(axis=1)
    assert (actual == y_pred).all()
    # fit_predict
    y_pred = model.fit_predict(X)
    precision = np.sum(y == y_pred)/n_samples
    if precision < 0.5:
        y_pred = np.mod(y_pred+1, 2)
    precision = np.sum(y == y_pred)/n_samples
    assert precision >= 0.9

    # multiple threads
    model = K_means(
        feature,
        n_clusters=2,
        max_iter=100,
        n_init=10,
        n_jobs=-1,
        tol=1e-4,
        verbose=False
    )
    model.fit(X)
    y_pred = model.predict(X)
    precision = np.sum(y == y_pred)/n_samples
    if precision < 0.5:
        y_pred = np.mod(y_pred+1, 2)
    precision = np.sum(y == y_pred)/n_samples
    assert precision >= 0.9
    y_pred = model.fit_predict(X)
    precision = np.sum(y == y_pred)/n_samples
    if precision < 0.5:
        y_pred = np.mod(y_pred+1, 2)
    precision = np.sum(y == y_pred)/n_samples
    assert precision >= 0.9
