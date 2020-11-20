import numpy as np
from numpy import random as rnd
import numpy.testing as np_test

from pyCovariance.features import intensity_euclidean, \
        mean_pixel_euclidean, \
        pixel_euclidean
from pyCovariance.features.base import _FeatureArray


def test_real_pixel_euclidean():
    p = 5
    N = 100
    N_mean = 10
    pix = pixel_euclidean(p)
    assert type(str(pix)) is str

    # test estimation
    X = rnd.randn(p, N)
    est = pix.estimation(X).export()
    assert est.dtype == np.float64
    np_test.assert_equal(est, X[:, int(N/2)+1])

    # test distance
    X1 = pix.estimation(rnd.randn(p, N))
    X2 = pix.estimation(rnd.randn(p, N))
    d1 = pix.distance(X1, X2)
    assert d1.dtype == np.float64
    d2 = np.linalg.norm(X1.export()-X2.export())
    np_test.assert_almost_equal(d1, d2)

    # test mean
    X = _FeatureArray((p,))
    for _ in range(N_mean):
        X.append(pix.estimation(rnd.randn(p, N)))
    m1 = pix.mean(X).export()
    assert m1.dtype == np.float64
    m2 = np.mean(X.export(), axis=0)
    np_test.assert_almost_equal(m1, m2)


def test_real_mean_pixel_euclidean():
    p = 5
    N = 100
    N_mean = 10
    pix = mean_pixel_euclidean(p)
    assert type(str(pix)) is str

    # test estimation
    X = rnd.randn(p, N)
    est = pix.estimation(X).export()
    assert est.dtype == np.float64
    feature = np.mean(X, axis=1)
    np_test.assert_equal(est, feature)

    # test distance
    X1 = pix.estimation(rnd.randn(p, N))
    X2 = pix.estimation(rnd.randn(p, N))
    d1 = pix.distance(X1, X2)
    assert d1.dtype == np.float64
    d2 = np.linalg.norm(X1.export()-X2.export())
    np_test.assert_almost_equal(d1, d2)

    # test mean
    X = _FeatureArray((p,))
    for _ in range(N_mean):
        X.append(pix.estimation(rnd.randn(p, N)))
    m1 = pix.mean(X).export()
    assert m1.dtype == np.float64
    m2 = np.mean(X.export(), axis=0)
    np_test.assert_almost_equal(m1, m2)


def test_real_intensity_euclidean():
    p = 5
    N = 100
    N_mean = 10
    inten = intensity_euclidean()
    assert type(str(inten)) is str

    # test estimation
    X = rnd.randn(p, N)
    feature = np.linalg.norm(X[:, int(N/2)+1])
    est = inten.estimation(X).export()
    assert est.dtype == np.float64
    np_test.assert_equal(est, feature)

    # test distance
    X1 = inten.estimation(rnd.randn(p, N))
    X2 = inten.estimation(rnd.randn(p, N))
    d1 = inten.distance(X1, X2)
    assert d1.dtype == np.float64
    d2 = np.linalg.norm(X1.export()-X2.export())
    np_test.assert_almost_equal(d1, d2)

    # test mean
    X = _FeatureArray((1,))
    for _ in range(N_mean):
        X.append(inten.estimation(rnd.randn(p, N)))
    m1 = inten.mean(X).export()
    assert m1.dtype == np.float64
    m2 = np.mean(X.export(), axis=0)
    np_test.assert_almost_equal(m1, m2)
