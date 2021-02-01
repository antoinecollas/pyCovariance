import numpy as np
from numpy import random as rnd
import numpy.linalg as la
import numpy.testing as np_test

from pyCovariance.features import\
        center_euclidean,\
        center_intensity_euclidean,\
        identity_euclidean,\
        intensity_vector_euclidean,\
        mean_vector_euclidean

from pyCovariance.features.base import _FeatureArray


def test_real_identity_euclidean():
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
    np_test.assert_equal(est, X)

    # test distance
    X1 = dummy.estimation(rnd.randn(p, N))
    X2 = dummy.estimation(rnd.randn(p, N))
    d1 = dummy.distance(X1, X2)
    assert d1.dtype == np.float64
    d2 = la.norm(X1.export()-X2.export())
    np_test.assert_almost_equal(d1, d2)

    # test mean
    X = _FeatureArray((p, N))
    for _ in range(N_mean):
        X.append(dummy.estimation(rnd.randn(p, N)))
    m1 = dummy.mean(X).export()
    assert m1.dtype == np.float64
    m2 = np.mean(X.export(), axis=0)
    np_test.assert_almost_equal(m1, m2)


def test_real_center_euclidean():
    p = 5
    N = 100
    N_mean = 10
    dummy = center_euclidean()(p, N)
    assert type(str(dummy)) is str

    # test estimation
    X = rnd.randn(p, N)
    est = dummy.estimation(X).export()
    assert est.dtype == np.float64
    np_test.assert_equal(est, X[:, int(N/2)])

    # test distance
    X1 = dummy.estimation(rnd.randn(p, N))
    X2 = dummy.estimation(rnd.randn(p, N))
    d1 = dummy.distance(X1, X2)
    assert d1.dtype == np.float64
    d2 = la.norm(X1.export()-X2.export())
    np_test.assert_almost_equal(d1, d2)

    # test mean
    X = _FeatureArray((p,))
    for _ in range(N_mean):
        X.append(dummy.estimation(rnd.randn(p, N)))
    m1 = dummy.mean(X).export()
    assert m1.dtype == np.float64
    m2 = np.mean(X.export(), axis=0)
    np_test.assert_almost_equal(m1, m2)


def test_real_center_intensity_euclidean():
    p = 5
    N = 100
    N_mean = 10
    inten = center_intensity_euclidean()(p, N)
    assert type(str(inten)) is str

    # test estimation
    X = rnd.randn(p, N)
    feature = la.norm(X[:, int(N/2)])
    est = inten.estimation(X).export()
    assert est.dtype == np.float64
    np_test.assert_equal(est, feature)

    # test distance
    X1 = inten.estimation(rnd.randn(p, N))
    X2 = inten.estimation(rnd.randn(p, N))
    d1 = inten.distance(X1, X2)
    assert d1.dtype == np.float64
    d2 = la.norm(X1.export()-X2.export())
    np_test.assert_almost_equal(d1, d2)

    # test mean
    X = _FeatureArray((1,))
    for _ in range(N_mean):
        X.append(inten.estimation(rnd.randn(p, N)))
    m1 = inten.mean(X).export()
    assert m1.dtype == np.float64
    m2 = np.mean(X.export(), axis=0)
    np_test.assert_almost_equal(m1, m2)


def test_real_mean_vector_euclidean():
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
    np_test.assert_equal(est, feature)

    # test distance
    X1 = dummy.estimation(rnd.randn(p, N))
    X2 = dummy.estimation(rnd.randn(p, N))
    d1 = dummy.distance(X1, X2)
    assert d1.dtype == np.float64
    d2 = la.norm(X1.export()-X2.export())
    np_test.assert_almost_equal(d1, d2)

    # test mean
    X = _FeatureArray((p,))
    for _ in range(N_mean):
        X.append(dummy.estimation(rnd.randn(p, N)))
    m1 = dummy.mean(X).export()
    assert m1.dtype == np.float64
    m2 = np.mean(X.export(), axis=0)
    np_test.assert_almost_equal(m1, m2)


def test_real_intensity_vector_euclidean():
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
    np_test.assert_equal(est, feature)

    # test distance
    X1 = inten.estimation(rnd.randn(p, N))
    X2 = inten.estimation(rnd.randn(p, N))
    d1 = inten.distance(X1, X2)
    assert d1.dtype == np.float64
    d2 = la.norm(X1.export()-X2.export())
    np_test.assert_almost_equal(d1, d2)

    # test mean
    X = _FeatureArray((N,))
    for _ in range(N_mean):
        X.append(inten.estimation(rnd.randn(p, N)))
    m1 = inten.mean(X).export()
    assert m1.dtype == np.float64
    m2 = np.mean(X.export(), axis=0)
    np_test.assert_almost_equal(m1, m2)
