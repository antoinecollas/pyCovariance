import autograd.numpy as np
import autograd.numpy.random as rnd
import numpy.testing as np_test

from pyCovariance.features.base import _FeatureArray


def test_FeatureArray():
    rnd.seed(123)

    p = 5
    N = 100

    # tests on a single manifold of vectors
    a = _FeatureArray((p,))

    # test dtype
    assert a.dtype == tuple()

    # test shape
    assert len(a) == 0
    assert a.shape == 0

    # append np array of size (p,)
    temp = rnd.randn(N, p)
    for i in range(N):
        a.append(temp[i])

    # test dtype
    assert a.dtype == (np.float64, )

    # test shape
    assert len(a) == N
    assert a.shape == ((N, p),)

    # test str
    assert type(str(a)) is str

    # test get_item and values stored
    for i in range(N):
        np_test.assert_equal(a[i].export(), temp[i])

    a = _FeatureArray((p,))

    # append np array of size (N, p)
    temp = rnd.randn(N, p)
    a.append(temp)

    # test dtype
    assert a.dtype == (np.float64, )

    # test shape
    assert a.nb_manifolds == 1
    assert len(a) == N
    assert a.shape == ((N, p),)

    # test str
    assert type(str(a)) is str

    # test values stored
    np_test.assert_equal(a.export(), temp)

    # tests on a single manifold of matrices
    a = _FeatureArray((p, p))

    # test dtype
    assert a.dtype == tuple()

    # test shape
    assert a.nb_manifolds == 1
    assert len(a) == 0
    assert a.shape == 0

    # append np array of size (p,p)
    temp = rnd.randn(N, p, p)
    for i in range(N):
        a.append(temp[i])

    # test dtype
    assert a.dtype == (np.float64, )

    # test shape
    assert len(a) == N
    assert a.shape == ((N, p, p),)

    # test str
    assert type(str(a)) is str

    # test get_item and values stored
    for i in range(N):
        np_test.assert_equal(a[i].export(), temp[i])

    # tests on product manifolds of matrices
    p1 = 5
    p2 = 7
    a1 = _FeatureArray((p1, p1), (p2, p2))

    # test dtype
    assert a1.dtype == tuple()

    # test shape
    assert a1.nb_manifolds == 2
    assert len(a1) == 0
    assert a1.shape == 0

    # append np array of size (p, p)
    temp1 = rnd.randn(N, p1, p1)
    temp2 = rnd.randn(N, p2, p2)
    for i in range(N):
        a1.append([temp1[i], temp2[i]])

    # test dtype
    assert a1.dtype == (np.float64, np.float64)

    # test shape
    assert len(a1) == N
    assert a1.shape == ((N, p1, p1), (N, p2, p2))

    # test str
    assert type(str(a1)) is str

    # test get_item and values stored
    for i in range(N):
        np_test.assert_equal(a1[i].export(), [temp1[i], temp2[i]])

    p1 = 5
    p2 = 7
    a2 = _FeatureArray((p1, p1), (p2, p2))

    # test dtype
    assert a2.dtype == tuple()

    # test shape
    assert a2.nb_manifolds == 2
    assert len(a2) == 0
    assert a2.shape == 0

    # append np array of size (p, p)
    temp21 = rnd.randn(N, p1, p1)
    temp22 = rnd.randn(N, p2, p2)
    a2.append([temp21, temp22])

    # test dtype
    assert a2.dtype == (np.float64, np.float64)

    # test shape
    assert len(a2) == N
    assert a2.shape == ((N, p1, p1), (N, p2, p2))

    # test str
    assert type(str(a2)) is str

    # test get_item and values stored
    np_test.assert_equal(a2.export(), [temp21, temp22])

    # add a2 to a1
    a1.append(a2)

    # check values stored
    for i in range(N):
        np_test.assert_equal(a1[i].export(), [temp1[i], temp2[i]])
    for i in range(N):
        np_test.assert_equal(a1[i+N].export(), [temp21[i], temp22[i]])
