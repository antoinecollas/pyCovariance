import autograd.numpy as np
from autograd.numpy import random as rnd
from pymanopt.manifolds import\
        ComplexGrassmann
from pymanopt.manifolds.product import _ProductTangentVector

from pyCovariance.manifolds import\
        ComplexRobustSubspaceIG,\
        StrictlyPositiveVectors
from pyCovariance.matrix_operators import multihconj, multiprod
from pyCovariance.testing import assert_allclose


class TestSingleComplexRobustSubspaceIGManifold():
    def __init__(self):
        self._n = n = 100
        self._p = p = 15
        self._rank = rank = 5
        self.man = ComplexRobustSubspaceIG(n=n, p=p, rank=rank)

    def check_type_dim(self, x):
        n, p, rank = self._n, self._p, self._rank
        assert len(x) == 2
        assert x[0].dtype == np.float64
        assert x[0].shape == (n, 1)
        assert x[1].dtype == np.complex128
        assert x[1].shape == (p, rank)

    def check_man(self, x):
        rank = self._rank

        assert type(x) == list
        self.check_type_dim(x)

        # check positivity
        assert (x[0] > 0).all()

        # check orthogonality
        assert_allclose(multiprod(multihconj(x[1]), x[1]), np.eye(rank))

    def check_tangent(self, x, u):
        rank = self._rank

        self.check_man(x)
        assert type(u) == _ProductTangentVector
        self.check_type_dim(u)

        # check in horizontal space
        assert_allclose(
            multiprod(multihconj(x[1]), u[1]),
            np.zeros((rank, rank), dtype=np.complex128)
        )

    # def test_dim(self):
    #     man = self.man
    #     p = self._p
    #     dim = 2*p + p*p
    #     assert man.dim == dim

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
        assert_allclose(self.man.inner(x, u, u), 1)

    def test_zerovec(self):
        rnd.seed(123)

        n, p, rank = self._n, self._p, self._rank
        man = self.man
        x = man.rand()
        u = man.zerovec(x)

        self.check_tangent(x, u)
        assert (u[0] == np.zeros((n, 1))).all()
        assert (u[1] == np.zeros((p, rank))).all()

    def test_inner(self):
        rnd.seed(123)

        n, p, rank = self._n, self._p, self._rank

        man = self.man
        x = man.rand()
        u = man.randvec(x)
        v = man.randvec(x)
        self.check_man(x)
        self.check_tangent(x, u)
        self.check_tangent(x, v)

        # textures
        man_tau = StrictlyPositiveVectors(n)
        desired_inner = rank * man_tau.inner(1+x[0], u[0], v[0])

        # subspace
        gr = ComplexGrassmann(p, rank)
        tmp = 2 * np.sum((x[0]**2)/(1+x[0]))
        tmp = tmp * gr.inner(x[1], u[1], v[1])
        desired_inner = desired_inner + tmp

        assert type(man.inner(x, u, v)) == np.float64
        assert_allclose(man.inner(x, u, v), desired_inner)

    def test_norm(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        u = man.randvec(x)
        norm = np.abs(rnd.normal(loc=2, scale=3))
        u = u * norm
        assert_allclose(man.norm(x, u), norm)

        assert type(man.norm(x, u)) == np.float64

    def test_proj(self):
        rnd.seed(123)

        n, p, rank = self._n, self._p, self._rank
        man = self.man
        x = man.rand()
        u = list()
        u.append(rnd.normal(size=(n, 1)))
        u.append(rnd.normal(size=(p, rank)) + 1j*rnd.normal(size=(p, rank)))
        u = man.proj(x, u)

        self.check_tangent(x, u)
        assert_allclose(u[0], man.proj(x, u)[0])
        assert_allclose(u[1], man.proj(x, u)[1])

    # def test_egrad2rgrad(self):
    #     rnd.seed(123)

    #     p = self._p
    #     man = self.man
    #     x = man.rand()
    #     egrad = list()
    #     egrad.append(rnd.normal(size=(p, 1)) + 1j*rnd.normal(size=(p, 1)))
    #     egrad.append(rnd.normal(size=(p, p)) + 1j*rnd.normal(size=(p, p)))
    #     grad = man.egrad2rgrad(x, egrad)
    #     self.check_tangent(x, grad)
    #     assert_allclose(0.5*x[1]@egrad[0], grad[0])
    #     assert_allclose(multiherm(x[1]@egrad[1]@x[1]), grad[1])

    def test_transp(self):
        rnd.seed(123)

        man = self.man
        x = man.rand()
        y = man.rand()
        u = man.randvec(x)
        t_u = man.transp(x, y, u)

        self.check_tangent(y, t_u)
