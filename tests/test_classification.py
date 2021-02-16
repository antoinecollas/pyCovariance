import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd
import numpy.testing as np_test
import os
from sklearn.metrics import accuracy_score

from pyCovariance.features.base import _FeatureArray
from pyCovariance.classification import\
        _estimate_features,\
        _compute_means,\
        _compute_pairwise_distances,\
        MDM,\
        TSclassifier
from pyCovariance.features import\
        center_euclidean,\
        mean_vector_euclidean
from pyCovariance.generation_data import\
        generate_covariance,\
        sample_normal_distribution


def test__estimate_features():
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


def helper_test_MDM(clf):
    n_classes = 5
    size_class = 100
    n_datasets = 3
    n_samples = size_class * n_classes

    for j in range(n_datasets):
        # test several p, N
        N = 20 + 5*j
        p = 3 + j

        X = np.zeros((n_samples, p, N))

        y = list()
        for k in range(n_classes):
            cov = generate_covariance(p)
            mu = rnd.randn(p, 1)
            for i in range(size_class):
                X[i + k*size_class] = sample_normal_distribution(N, cov) + mu

            y.append(k*np.ones(size_class))
        y = np.concatenate(y)
        assert X.shape[0] == len(y)

        idx = rnd.permutation(n_samples)

        # generating training data
        X_train = X[idx[:n_samples//2]]
        y_train = y[idx[:n_samples//2]]

        # generating test data
        X_test = X[idx[n_samples//2:]]
        y_test = y[idx[n_samples//2:]]

        # fit + predict on train set
        clf.fit(X_train, y_train)
        means = clf._means
        for i, mean in enumerate(means):
            temp = X_train[y_train == i]
            temp = np.mean(temp, axis=2)
            temp = np.mean(temp, axis=0)
            np_test.assert_almost_equal(mean.export(), temp)
        y = clf.predict(X_train)
        assert accuracy_score(y_train, y) > 0.95

        # fit + predict on test set
        y = clf.predict(X_test)
        assert accuracy_score(y_test, y) > 0.95

        # fit + predict on test set
        y = clf.transform(X_test)
        y = y.argmin(axis=1)
        assert accuracy_score(y_test, y) > 0.95

        # predict proba
        proba = clf.predict_proba(X_test)
        assert (proba >= 0).all()
        actual = np.sum(proba, axis=1)
        desired = np.ones(len(X_test))
        assert ((actual - desired) < 1e-8).all()
        actual = proba.argmax(axis=1)
        assert (actual == y).all()

        # fit_predict
        y = clf.fit_predict(X_train, y_train)
        assert accuracy_score(y_train, y) > 0.95


def test_MDM_single_job():
    feature = mean_vector_euclidean()
    mdm = MDM(feature, n_jobs=1)
    helper_test_MDM(mdm)


def test_MDM_multiple_jobs():
    feature = mean_vector_euclidean()
    mdm = MDM(feature, n_jobs=-1)
    helper_test_MDM(mdm)


def helper_test_classifier(clf):
    n_classes = 5
    size_class = 100
    n_datasets = 3
    n_samples = size_class * n_classes

    for j in range(n_datasets):
        # test several p, N
        N = 20 + 5*j
        p = 3 + 2*j

        X = np.zeros((n_samples, p, N))

        y = list()
        for k in range(n_classes):
            cov = generate_covariance(p)
            mu = rnd.randn(p, 1)
            for i in range(size_class):
                X[i + k*size_class] = sample_normal_distribution(N, cov) + mu

            y.append(k*np.ones(size_class))
        y = np.concatenate(y)
        assert X.shape[0] == len(y)

        idx = rnd.permutation(n_samples)

        # generating training data
        X_train = X[idx[:n_samples//2]]
        y_train = y[idx[:n_samples//2]]

        # generating test data
        X_test = X[idx[n_samples//2:]]
        y_test = y[idx[n_samples//2:]]

        # fit + predict on train set
        clf.fit(X_train, y_train)
        mean = clf._mean
        temp = np.mean(X_train, axis=2)
        temp = np.mean(temp, axis=0)
        np_test.assert_almost_equal(mean.export(), temp)
        y = clf.predict(X_train)
        assert accuracy_score(y_train, y) > 0.95

        # predict on test set
        y = clf.predict(X_test)
        assert accuracy_score(y_test, y) > 0.95

        # predict proba
        proba = clf.predict_proba(X_test)
        assert (proba >= 0).all()
        actual = np.sum(proba, axis=1)
        desired = np.ones(len(X_test))
        assert ((actual - desired) < 1e-8).all()
        actual = proba.argmax(axis=1)
        assert (actual == y).all()

        # fit_predict
        y = clf.fit_predict(X_train, y_train)
        assert accuracy_score(y_train, y) > 0.95


def test_TSclassifier_single_job():
    feature = mean_vector_euclidean()
    clf = TSclassifier(feature, n_jobs=1)
    helper_test_classifier(clf)


def test_TSclassifier_multiple_job():
    feature = mean_vector_euclidean()
    clf = TSclassifier(feature, n_jobs=-1)
    helper_test_classifier(clf)
