import autograd.numpy as np
from autograd.numpy import linalg as la, random as rnd

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold


class StrictlyPositiveVectors(EuclideanEmbeddedSubmanifold):
    """Manifold of k strictly positive n-dimensional vectors: ((R++)^n)^k.
    Since ((R++)^n)^k is isomorphic to
    (D_n^{++})^k (manifold of positive definite diagonal matrices of size n),
    the geometry is inherited of the positive definite matrices.
    """
    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        if k == 1:
            name = ("Manifold of strictly positive vectors of size {}").format(
                n)
        else:
            name = ("Product manifold of {} \
                    strictly positive vectors of size {}").format(k, n)
        dimension = int(k * n)
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def inner(self, x, u, v):
        inv_x = (1./x)
        return np.sum(inv_x*u*inv_x*v)

    def proj(self, x, u):
        return u

    def norm(self, x, u):
        return np.sqrt(self.inner(x, u, u))

    def rand(self):
        k, n = self._k, self._n
        if k == 1:
            x = rnd.uniform(low=1e-6, high=1, size=(n, 1))
        else:
            x = rnd.uniform(low=1e-6, high=1, size=(k, n, 1))
        return x

    def randvec(self, x):
        k, n = self._k, self._n
        if k == 1:
            u = rnd.normal(size=(n, 1))
        else:
            u = rnd.normal(size=(k, n, 1))
        return u / self.norm(x, u)

    def zerovec(self, x):
        k, n = self._k, self._n
        if k == 1:
            u = np.zeros((n, 1))
        else:
            u = np.zeros((k, n, 1))
        return u

    def dist(self, x, y):
        return la.norm(np.log(x)-np.log(y))

    def egrad2rgrad(self, x, u):
        return u*(x**2)

    def ehess2rhess(self, x, egrad, ehess, u):
        return ehess*(x**2) + egrad*u*x

    def exp(self, x, u):
        return x*np.exp((1./x)*u)

    # order 2 retraction
    def retr(self, x, u):
        tmp = x + u + (1/2)*(x**-1)*(u**2)
        return tmp

    def log(self, x, y):
        return x*np.log((1./x)*y)

    def transp(self, x1, x2, d):
        res = self.proj(x2, x2*(x1**(-1)*d))
        return res


class SpecialStrictlyPositiveVectors(EuclideanEmbeddedSubmanifold):
    """Manifold of k special strictly positive n-dimensional
    vectors, denoted (S(R++)^n)^k.
    Hence, the product of the elements of one vector is equal to 1.
    """
    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        if k == 1:
            name = ("Manifold of special strictly \
                    positive vectors of size {}").format(n)
        else:
            name = ("Product manifold of {} \
                    special strictly positive \
                    vectors of size {}").format(k, n)
        dimension = int(k * (n - 1))
        self.Rpos = StrictlyPositiveVectors(n, k)
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return self.Rpos.typicaldist()

    def _to_unit_prod(self, x):
        k, n = self._k, self._n
        if k == 1:
            x = x / np.exp((1/n)*np.sum(np.log(x), axis=0, keepdims=True))
        else:
            x = x / np.exp((1/n)*np.sum(np.log(x), axis=1, keepdims=True))
        return x

    def inner(self, x, u, v):
        return self.Rpos.inner(x, u, v)

    def proj(self, x, u):
        k, n = self._k, self._n
        if k == 1:
            u = u - (1/n) * np.sum(u/x) * x
        else:
            u = u - (1/n) * np.sum(u/x, axis=1, keepdims=True) * x
        return u

    def norm(self, x, u):
        return self.Rpos.norm(x, u)

    def rand(self):
        x = self.Rpos.rand()
        x = self._to_unit_prod(x)
        return x

    def randvec(self, x):
        u = self.Rpos.randvec(x)
        u = self.proj(x, u)
        return u / self.norm(x, u)

    def zerovec(self, x):
        return self.Rpos.zerovec(x)

    def dist(self, x, y):
        return self.Rpos.dist(x, y)

    def egrad2rgrad(self, x, u):
        return self.proj(x, self.Rpos.egrad2rgrad(x, u))

    # def ehess2rhess(self, x, egrad, ehess, u):

    def exp(self, x, u):
        y = self.Rpos.exp(x, u)
        # for numerical stability
        y = self._to_unit_prod(y)
        return y

    # order 2 retraction
    def retr(self, x, u):
        y = x + u + (1/2)*(x**-1)*(u**2)
        y = self._to_unit_prod(y)
        return y

    def log(self, x, y):
        return x*np.log((1./x)*y)

    def transp(self, x1, x2, d):
        return self.proj(x2, self.Rpos.transp(x1, x2, d))
