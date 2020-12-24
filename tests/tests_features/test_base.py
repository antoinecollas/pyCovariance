import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import numpy.testing as np_test
from pymanopt.manifolds.euclidean import Symmetric, Euclidean

from pyCovariance.features.base import _FeatureArray, Product


def test_FeatureArray():
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

    # test rmul
    c = 3
    shape_a1 = a1.shape
    dtype_a1 = a1.dtype
    a = c*a1
    shape_a = a.shape
    dtype_a = a.dtype
    assert shape_a == shape_a1
    assert dtype_a == dtype_a1

    # check values stored
    for i in range(N):
        np_test.assert_equal(a[i].export(), [c*temp1[i], c*temp2[i]])
    for i in range(N):
        np_test.assert_equal(a[i+N].export(), [c*temp21[i], c*temp22[i]])

    # test mul
    c = 4
    shape_a1 = a1.shape
    a = a1*c
    shape_a = a.shape
    assert shape_a == shape_a1

    # check values stored
    for i in range(N):
        np_test.assert_equal(a[i].export(), [c*temp1[i], c*temp2[i]])
    for i in range(N):
        np_test.assert_equal(a[i+N].export(), [c*temp21[i], c*temp22[i]])


def test_product():
    p1 = 10
    p2 = 8
    p3 = 3
    w = (2, 3)
    m = Product([Symmetric(p1), Euclidean(p2, p3)], w)

    # rand
    X = m.rand()
    assert len(X) == 2
    assert X[0].shape == (p1, p1)
    assert X[1].shape == (p2, p3)
    np_test.assert_almost_equal(X[0], 1/2 * (X[0] + X[0].T))

    # randvec
    xi = m.randvec(X)
    assert len(xi) == 2
    assert xi[0].shape == (p1, p1)
    assert xi[1].shape == (p2, p3)
    np_test.assert_almost_equal(xi[0], 1/2 * (xi[0] + xi[0].T))

    # zerovec
    xi = m.zerovec(X)
    assert len(xi) == 2
    np_test.assert_almost_equal(xi[0], np.zeros((p1, p1)))
    np_test.assert_almost_equal(xi[1], np.zeros((p2, p3)))

    # inner
    xi = m.randvec(X)
    eta = m.randvec(X)
    res = w[0] * np.trace(xi[0].T@eta[0]) + w[1] * np.trace(xi[1].T@eta[1])
    np_test.assert_almost_equal(m.inner(X, xi, eta), res)

    # norm
    res = w[0] * np.trace(xi[0].T@xi[0]) + w[1] * np.trace(xi[1].T@xi[1])
    res = np.sqrt(res)
    np_test.assert_almost_equal(m.norm(X, xi), res)

    # dist
    X1 = m.rand()
    X2 = m.rand()
    d = w[0]*(la.norm(X1[0]-X2[0])**2) + w[1]*(la.norm(X1[1]-X2[1])**2)
    d = np.sqrt(d)
    np_test.assert_almost_equal(m.dist(X1, X2), d)

    # proj
    xi = [rnd.randn(p1, p1), rnd.randn(p2, p3)]
    p = m.proj(X, xi)
    assert len(p) == 2
    np_test.assert_almost_equal(p[0], 1/2 * (xi[0] + xi[0].T))
    np_test.assert_almost_equal(p[1], xi[1])

    # egrad2rgrad
    xi = [rnd.randn(p1, p1), rnd.randn(p2, p3)]
    p = m.proj(X, xi)
    res = [(1/w[0])*p[0], (1/w[1])*p[1]]
    grad = m.egrad2rgrad(X, xi)
    np_test.assert_almost_equal(grad[0], res[0])
    np_test.assert_almost_equal(grad[1], res[1])

    # exp
    X = m.rand()
    xi = m.randvec(X)
    res = [X[0] + xi[0], X[1] + xi[1]]
    exp = m.exp(X, xi)
    np_test.assert_almost_equal(exp[0], res[0])
    np_test.assert_almost_equal(exp[1], res[1])

    # retr
    X = m.rand()
    xi = m.randvec(X)
    res = [X[0] + xi[0], X[1] + xi[1]]
    retr = m.retr(X, xi)
    np_test.assert_almost_equal(retr[0], res[0])
    np_test.assert_almost_equal(retr[1], res[1])

    # log
    X1 = m.rand()
    X2 = m.rand()
    log = m.log(X1, X2)
    res = [X2[0] - X1[0], X2[1] - X1[1]]
    np_test.assert_almost_equal(log[0], res[0])
    np_test.assert_almost_equal(log[1], res[1])

    # transp
    xi = m.randvec(X1)
    transp = m.transp(X1, X2, xi)
    np_test.assert_almost_equal(transp[0], xi[0])
    np_test.assert_almost_equal(transp[1], xi[1])

    # pairmean
    X1 = m.rand()
    X2 = m.rand()
    pairmean = m.pairmean(X1, X2)
    res = [(X1[0] + X2[0]) / 2, (X1[1] + X2[1]) / 2]
    np_test.assert_almost_equal(pairmean[0], res[0])
    np_test.assert_almost_equal(pairmean[1], res[1])
