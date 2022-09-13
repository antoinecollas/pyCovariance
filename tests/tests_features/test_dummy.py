import numpy as np
import autograd.numpy.linalg as la
from autograd.numpy import random as rnd

from pyCovariance.features import\
        center_euclidean,\
        center_intensity_euclidean,\
        identity_euclidean,\
        intensity_vector_euclidean,\
        mean_vector_euclidean
from pyCovariance.features.base import _FeatureArray
from pyCovariance.testing import assert_allclose


def test_real_identity_euclidean():
    rnd.seed(123)

    p = 5
    N = 100
    N_mean = 10
    dummy = identity_euclidean()(p, N)
    assert type(str(dummy)) is str

    # test estimation
    X = rnd.randn(p, N)
    est = dummy.estimation(X).export()
    assert est.dtype == np.float64
    assert est.shape == (p, N)
    assert_allclose(est, X)

    # test distance
    X1 = dummy.estimation(rnd.randn(p, N))
    X2 = dummy.estimation(rnd.randn(p, N))
    d1 = dummy.distance(X1, X2)
    assert d1.dtype == np.float64
    d2 = la.norm(X1.export()-X2.export())
    assert_allclose(d1, d2)

    # test log
    batch_size = 10
    X = dummy.estimation(rnd.randn(p, N))
    Y = dummy.estimation(rnd.randn(p, N))
    for _ in range(batch_size-1):
        Y.append(rnd.randn(p, N))
    log = dummy.log(X, Y)
    assert type(log) == _FeatureArray
    assert len(log) == batch_size
    log = log.export()
    assert type(log) == np.ndarray
    assert log.dtype == np.float64
    assert log.shape == (batch_size, p, N)
    X = X.export()
    Y = Y.export()
    for i in range(batch_size):
        assert_allclose(log[i], Y[i]-X)

    # test vectorize log
    batch_size = 10
    X = dummy.estimation(rnd.randn(p, N))
    Y = dummy.estimation(rnd.randn(p, N))
    for _ in range(batch_size-1):
        Y.append(rnd.randn(p, N))
    log = dummy.log(X, Y, vectorize=True)
    assert type(log) == np.ndarray
    assert log.dtype == np.float64
    assert len(log) == batch_size
    assert log.shape == (batch_size, p*N)
    X = X.export()
    Y = Y.export()
    for i in range(batch_size):
        assert_allclose(log[i], (Y[i]-X).reshape(-1))

    # test mean
    X = _FeatureArray((p, N))
    for _ in range(N_mean):
        X.append(dummy.estimation(rnd.randn(p, N)))
    m1 = dummy.mean(X).export()
    assert m1.dtype == np.float64
    m2 = np.mean(X.export(), axis=0)
    assert_allclose(m1, m2)


def test_real_center_euclidean():
    rnd.seed(123)

    p = 5
    N = 100
    N_mean = 10
    dummy = center_euclidean()(p, N)
    assert type(str(dummy)) is str

    # test estimation
    X = rnd.randn(p, N)
    est = dummy.estimation(X).export()
    assert est.dtype == np.float64
    assert_allclose(est, X[:, int(N/2)])

    # test distance
    X1 = dummy.estimation(rnd.randn(p, N))
    X2 = dummy.estimation(rnd.randn(p, N))
    d1 = dummy.distance(X1, X2)
    assert d1.dtype == np.float64
    d2 = la.norm(X1.export()-X2.export())
    assert_allclose(d1, d2)

    # test log
    batch_size = 10
    X = dummy.estimation(rnd.randn(p, N))
    Y = dummy.estimation(rnd.randn(p, N))
    for _ in range(batch_size-1):
        Y.append(dummy.estimation(rnd.randn(p, N)))
    log = dummy.log(X, Y)
    assert type(log) == _FeatureArray
    assert len(log) == batch_size
    log = log.export()
    assert type(log) == np.ndarray
    assert log.dtype == np.float64
    assert log.shape == (batch_size, p)
    X = X.export()
    Y = Y.export()
    for i in range(batch_size):
        assert_allclose(log[i], Y[i]-X)

    # test vectorize log
    batch_size = 10
    X = dummy.estimation(rnd.randn(p, N))
    Y = dummy.estimation(rnd.randn(p, N))
    for _ in range(batch_size-1):
        Y.append(dummy.estimation(rnd.randn(p, N)))
    log = dummy.log(X, Y, vectorize=True)
    assert type(log) == np.ndarray
    assert log.dtype == np.float64
    assert len(log) == batch_size
    assert log.shape == (batch_size, p)
    X = X.export()
    Y = Y.export()
    for i in range(batch_size):
        assert_allclose(log[i], (Y[i]-X).reshape(-1))

    # test mean
    X = _FeatureArray((p,))
    for _ in range(N_mean):
        X.append(dummy.estimation(rnd.randn(p, N)))
    m1 = dummy.mean(X).export()
    assert m1.dtype == np.float64
    m2 = np.mean(X.export(), axis=0)
    assert_allclose(m1, m2)


def test_real_center_intensity_euclidean():
    rnd.seed(123)

    p = 5
    N = 100
    N_mean = 10
    dummy = center_intensity_euclidean()(p, N)
    assert type(str(dummy)) is str

    # test estimation
    X = rnd.randn(p, N)
    feature = la.norm(X[:, int(N/2)])
    est = dummy.estimation(X).export()
    assert est.dtype == np.float64
    assert_allclose(est, feature)

    # test distance
    X1 = dummy.estimation(rnd.randn(p, N))
    X2 = dummy.estimation(rnd.randn(p, N))
    d1 = dummy.distance(X1, X2)
    assert d1.dtype == np.float64
    d2 = la.norm(X1.export()-X2.export())
    assert_allclose(d1, d2)

    # test log
    batch_size = 10
    X = dummy.estimation(rnd.randn(p, N))
    Y = dummy.estimation(rnd.randn(p, N))
    for _ in range(batch_size-1):
        Y.append(dummy.estimation(rnd.randn(p, N)))
    log = dummy.log(X, Y)
    assert type(log) == _FeatureArray
    assert len(log) == batch_size
    log = log.export()
    assert type(log) == np.ndarray
    assert log.dtype == np.float64
    assert log.shape == (batch_size, 1)
    X = X.export()
    Y = Y.export()
    for i in range(batch_size):
        assert_allclose(log[i], Y[i]-X)

    # test vectorize log
    batch_size = 10
    X = dummy.estimation(rnd.randn(p, N))
    Y = dummy.estimation(rnd.randn(p, N))
    for _ in range(batch_size-1):
        Y.append(dummy.estimation(rnd.randn(p, N)))
    log = dummy.log(X, Y, vectorize=True)
    assert type(log) == np.ndarray
    assert log.dtype == np.float64
    assert len(log) == batch_size
    assert log.shape == (batch_size, 1)
    X = X.export()
    Y = Y.export()
    for i in range(batch_size):
        assert_allclose(log[i], (Y[i]-X).reshape(-1))

    # test mean
    X = _FeatureArray((1,))
    for _ in range(N_mean):
        X.append(dummy.estimation(rnd.randn(p, N)))
    m1 = dummy.mean(X).export()
    assert m1.dtype == np.float64
    m2 = np.mean(X.export(), axis=0)
    assert_allclose(m1, m2)


def test_real_mean_vector_euclidean():
    rnd.seed(123)

    p = 5
    N = 100
    N_mean = 10
    dummy = mean_vector_euclidean()(p, N)
    assert type(str(dummy)) is str

    # test estimation
    X = rnd.randn(p, N)
    est = dummy.estimation(X).export()
    assert est.dtype == np.float64
    feature = np.mean(X, axis=1)
    assert_allclose(est, feature)

    # test distance
    X1 = dummy.estimation(rnd.randn(p, N))
    X2 = dummy.estimation(rnd.randn(p, N))
    d1 = dummy.distance(X1, X2)
    assert d1.dtype == np.float64
    d2 = la.norm(X1.export()-X2.export())
    assert_allclose(d1, d2)

    # test mean
    X = _FeatureArray((p,))
    for _ in range(N_mean):
        X.append(dummy.estimation(rnd.randn(p, N)))
    m1 = dummy.mean(X).export()
    assert m1.dtype == np.float64
    m2 = np.mean(X.export(), axis=0)
    assert_allclose(m1, m2)


def test_real_intensity_vector_euclidean():
    rnd.seed(123)

    p = 5
    N = 100
    N_mean = 10
    inten = intensity_vector_euclidean()(p, N)
    assert type(str(inten)) is str

    # test estimation
    X = rnd.randn(p, N)
    feature = la.norm(X, axis=0)
    est = inten.estimation(X).export()
    assert est.dtype == np.float64
    assert est.shape == (N,)
    assert_allclose(est, feature)

    # test distance
    X1 = inten.estimation(rnd.randn(p, N))
    X2 = inten.estimation(rnd.randn(p, N))
    d1 = inten.distance(X1, X2)
    assert d1.dtype == np.float64
    d2 = la.norm(X1.export()-X2.export())
    assert_allclose(d1, d2)

    # test mean
    X = _FeatureArray((N,))
    for _ in range(N_mean):
        X.append(inten.estimation(rnd.randn(p, N)))
    m1 = inten.mean(X).export()
    assert m1.dtype == np.float64
    m2 = np.mean(X.export(), axis=0)
    assert_allclose(m1, m2)
