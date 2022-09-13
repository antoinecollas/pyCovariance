import autograd.numpy as np
from autograd.numpy import linalg as la
from numpy import random as rnd
from pymanopt.manifolds.product import _ProductTangentVector

from pyCovariance.manifolds import\
        ComplexCompoundGaussianMLConstrainedTexture
from pyCovariance.matrix_operators import multiherm
from pyCovariance.testing import assert_allclose


class TestProductComplexCompoundGaussianMLConstrainedTextureManifold():
    def __init__(self):
        self._p = p = 15
        self._n = n = 100
        self._k = k = 5
        self._weights = weights = (1.5, 7)
        self.man = ComplexCompoundGaussianMLConstrainedTexture(
            p=p, n=n, k=k, weights=weights)

    def check_type_dim(self, x):
        k, p, n = self._k, self._p, self._n
        assert len(x) == 3
        assert x[0].dtype == np.complex128
        assert x[0].shape == (k, p, 1)
        assert x[1].dtype == np.complex128
        assert x[1].shape == (k, p, p)
        assert x[2].dtype == np.float64
        assert x[2].shape == (k, n, 1)

    def check_man(self, theta):
        assert type(theta) == list
        self.check_type_dim(theta)

        # check symmetry
        assert_allclose(theta[1], multiherm(theta[1]))

        for i in range(self._k):
            x = [theta[0][:, i], theta[1][i, :, :], theta[2][i, :]]

            # check positivity
            assert (la.eigvalsh(x[1]) > 0).all()
            assert (x[2] > 0).all()

            # check textures with unit prod
            assert_allclose(np.prod(x[2]), 1)

    def check_tangent(self, x, u):
        self.check_man(x)
        assert type(u) == _ProductTangentVector
        self.check_type_dim(u)

        # check symmetry
        assert_allclose(x[1], multiherm(x[1]))

        # check null trace
        assert_allclose(np.sum((x[2]**-1)*u[2]), 0)

    def test_dim(self):
        man = self.man
        k, p, n = self._k, self._p, self._n
        dim = k * (2*p + p*p + n - 1)
        assert man.dim == dim

    def test_rand(self):
        rnd.seed(123)

        x = self.man.rand()

        self.check_man(x)

    def test_randvec(self):
        rnd.seed(123)

        # Just test that randvec returns an element of the tangent space
        # with norm 1
        man = self.man
        x = man.rand()
        u = man.randvec(x)

        self.check_tangent(x, u)
        assert_allclose(self.man.norm(x, u), 1)

    def test_zerovec(self):
        rnd.seed(123)

        p, n = self._p, self._n
        man = self.man
        x = man.rand()
        u = man.zerovec(x)

        self.check_tangent(x, u)
        assert (u[0] == np.zeros((p, 1))).all()
        assert (u[1] == np.zeros((p, p))).all()
        assert (u[2] == np.zeros((n, 1))).all()

    def test_inner(self):
        rnd.seed(123)

        k = self._k
        weights = self._weights
        man = self.man
        theta = man.rand()
        xi = man.randvec(theta)
        eta = man.randvec(theta)

        res = 0
        for i in range(k):
            x = [theta[0][i, :, :], theta[1][i, :, :], theta[2][:, i]]
            u = [xi[0][i, :, :], xi[1][i, :, :], xi[2][:, i]]
            v = [eta[0][i, :, :], eta[1][i, :, :], eta[2][:, i]]
            sigma_inv = la.inv(x[1])

            inner = 2 * weights[0] * np.real(u[0].conj().T@sigma_inv@v[0])
            inner += weights[0] * np.real(
                np.trace(sigma_inv@u[1]@sigma_inv@v[1]))
            inner += weights[1] * (u[2]*(x[2]**(-1))).T@(v[2]*(x[2]**(-1)))
            res += inner

        assert type(man.inner(x, u, v)) == np.float64
        assert_allclose(man.inner(x, u, v), inner)

    def test_norm(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        u = man.randvec(x)
        norm = np.abs(rnd.normal(loc=2, scale=3))
        u = u * norm

        assert type(man.norm(x, u)) == np.float64
        assert_allclose(man.norm(x, u), norm)

    def test_proj(self):
        rnd.seed(123)

        k, n, p = self._k, self._n, self._p
        man = self.man
        x = man.rand()
        u = list()
        u.append(rnd.normal(size=(k, p, 1)) + 1j*rnd.normal(size=(k, p, 1)))
        u.append(rnd.normal(size=(k, p, p)) + 1j*rnd.normal(size=(k, p, p)))
        u.append(rnd.normal(size=(k, n, 1)))
        u = man.proj(x, u)

        self.check_tangent(x, u)
        assert_allclose(u[0], man.proj(x, u)[0])
        assert_allclose(u[1], man.proj(x, u)[1])
        assert_allclose(u[2], man.proj(x, u)[2])

    def test_egrad2rgrad(self):
        rnd.seed(123)

        k, n, p = self._k, self._n, self._p
        weights = self._weights
        man = self.man
        x = man.rand()
        egrad = list()
        egrad.append(
            rnd.normal(size=(k, p, 1)) + 1j*rnd.normal(size=(k, p, 1)))
        egrad.append(
            rnd.normal(size=(k, p, p)) + 1j*rnd.normal(size=(k, p, p)))
        egrad.append(rnd.normal(size=(k, n, 1)))
        self.check_tangent(x, man.egrad2rgrad(x, egrad))

        grad = [
            np.zeros_like(egrad[0]),
            np.zeros_like(egrad[1]),
            np.zeros_like(egrad[2])
        ]
        for i in range(k):
            sigma = x[1][i, :, :]
            grad[0][i, :] = (1/(2*weights[0])) * sigma@egrad[0][i, :]
            tmp = sigma@multiherm(egrad[1][i, :, :])@sigma
            grad[1][i, :, :] = (1/weights[0]) * tmp
            tmp = (1/weights[1]) * (x[2][i, :]**2)*egrad[2][i, :]
            c = np.sum(tmp*(x[2][i, :]**-1))
            grad[2][i, :] = tmp - (1/n) * c * x[2][i, :]

        assert_allclose(grad[0], man.egrad2rgrad(x, egrad)[0])
        assert_allclose(grad[1], man.egrad2rgrad(x, egrad)[1])
        assert_allclose(grad[2], man.egrad2rgrad(x, egrad)[2])

    def test_retr(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()

        # retr(x, 0) = x
        u = man.zerovec(x)
        r = man.retr(x, u)
        assert_allclose(r[0], x[0])
        assert_allclose(r[1], x[1])
        assert_allclose(r[2], x[2])

        # retr(x, u) = x + u.
        u = man.randvec(x)
        u = u * 1e-5
        r = man.retr(x, u)
        assert_allclose(r[0], x[0] + u[0])
        assert_allclose(r[1], x[1] + u[1])
        assert_allclose(r[2], x[2] + u[2])

        self.check_man(r)

    def test_transp(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        y = man.rand()
        u = man.randvec(x)
        t_u = man.transp(x, y, u)

        self.check_tangent(y, t_u)

    def test_div_orth(self):
        rnd.seed(123)

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        assert type(man.div_orth(x1, x2)) is np.float64

    def test_div_orth_sym(self):
        rnd.seed(123)

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        assert type(man.div_orth_sym(x1, x2)) is np.float64
        assert man.div_orth_sym(x1, x2) == man.div_orth_sym(x2, x1)

    def test_div_scale(self):
        rnd.seed(123)

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        assert type(man.div_scale(x1, x2)) is np.float64

    def test_div_scale_sym(self):
        rnd.seed(123)

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        assert type(man.div_scale_sym(x1, x2)) is np.float64
        assert man.div_scale_sym(x1, x2) == man.div_scale_sym(x2, x1)

    def test_dist(self):
        rnd.seed(123)

        man = self.man
        x1 = man.rand()
        x2 = man.rand()

        assert type(man.dist(x1, x2)) is np.float64
