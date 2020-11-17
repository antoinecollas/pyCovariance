import numpy as np
from numpy import random as rnd
import numpy.testing as np_test

from pyCovariance.features import intensity_euclidean, \
        mean_pixel_euclidean, \
        pixel_euclidean
from pyCovariance.features.base import _FeatureArray


def test_pixel_euclidean():
    p = 5
    N = 100
    N_mean = 10
    pix = pixel_euclidean(p)
    assert type(str(pix)) is str

    # test estimation
    X = rnd.randn(N, p)
    np_test.assert_equal(pix.estimation(X).export(), X[int(N/2)+1])

    # test distance
    X1 = pix.estimation(rnd.randn(N, p))
    X2 = pix.estimation(rnd.randn(N, p))
    d = np.linalg.norm(X1.export()-X2.export())
    np_test.assert_almost_equal(pix.distance(X1, X2), d)

    # test mean
    X = _FeatureArray((p,))
    for _ in range(N_mean):
        X.append(pix.estimation(rnd.randn(N, p)))
    m = np.mean(X.export(), axis=0)
    np_test.assert_almost_equal(pix.mean(X).export(), m)


def test_mean_pixel_euclidean():
    p = 5
    N = 100
    N_mean = 10
    pix = mean_pixel_euclidean(p)
    assert type(str(pix)) is str

    # test estimation
    X = rnd.randn(N, p)
    feature = np.mean(X, axis=0)
    np_test.assert_equal(pix.estimation(X).export(), feature)

    # test distance
    X1 = pix.estimation(rnd.randn(N, p))
    X2 = pix.estimation(rnd.randn(N, p))
    d = np.linalg.norm(X1.export()-X2.export())
    np_test.assert_almost_equal(pix.distance(X1, X2), d)

    # test mean
    X = _FeatureArray((p,))
    for _ in range(N_mean):
        X.append(pix.estimation(rnd.randn(N, p)))
    m = np.mean(X.export(), axis=0)
    np_test.assert_almost_equal(pix.mean(X).export(), m)


def test_intensity_euclidean():
    p = 5
    N = 100
    N_mean = 10
    inten = intensity_euclidean()
    assert type(str(inten)) is str

    # test estimation
    X = rnd.randn(N, p)
    feature = np.linalg.norm(X[int(N/2)+1])
    np_test.assert_equal(inten.estimation(X).export(), feature)

    # test distance
    X1 = inten.estimation(rnd.randn(N, p))
    X2 = inten.estimation(rnd.randn(N, p))
    d = np.linalg.norm(X1.export()-X2.export())
    np_test.assert_almost_equal(inten.distance(X1, X2), d)

    # test mean
    X = _FeatureArray((1,))
    for _ in range(N_mean):
        X.append(inten.estimation(rnd.randn(N, p)))
    m = np.mean(X.export())
    np_test.assert_almost_equal(inten.mean(X).export(), m)
