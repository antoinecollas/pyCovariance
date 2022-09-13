import autograd.numpy as np
from autograd.numpy import linalg as la

from pymanopt.manifolds import\
        ComplexGrassmann
from pymanopt.manifolds.manifold import Manifold
from pymanopt.manifolds.product import _ProductTangentVector

from .product import Product
from .strictly_positive_vectors import StrictlyPositiveVectors


EPS = 0
ITER_MAX = 20


class ComplexRobustSubspaceIG(Manifold):
    """Information geometry of complex and heteroscedastic signals
    embedded in white Gaussian noise (aka "tau UUH model").
    2 parameters: textures (strictly positive vector) and
    subspace (element of Grassmann manifold).
    """
    def __init__(self, n, p, rank, k=1):
        # n: number of data
        self._n = n
        # p: dimension of data
        self._p = p
        # rank: rank of subspace
        self._rank = rank
        # k: optimize several distributions at the same time
        self._k = k
        if k > 1:
            raise ValueError('Optimization of several \
                             distributions not implemented ...')
        if k == 1:
            name = ('Information geometry of \
                    complex robust subspace distributions \
                    aka "tau UUH model" \
                    ({} textures \
                    {} dimensional subspace.').format(n, rank)
        if k == 1:
            prod_manifold = Product([
                StrictlyPositiveVectors(n),
                ComplexGrassmann(p, rank)
            ])
        dimension = prod_manifold.dim
        self._prod_manifold = prod_manifold
        super().__init__(name, dimension, point_layout=2)

    def rand(self):
        return self._prod_manifold.rand()

    def randvec(self, x):
        u = self._prod_manifold.randvec(x)
        u = u * (1/self.norm(x, u))
        return u

    def zerovec(self, x):
        n, p, rank = self._n, self._p, self._rank
        tmp = list()
        tmp.append(np.zeros((n, 1)))
        tmp.append(np.zeros((p, rank), dtype=np.complex128))
        return _ProductTangentVector(tmp)

    def inner(self, x, u, v):
        rank = self._rank

        # textures
        res = rank * np.real((u[0]/(1+x[0])).T @ (v[0]/(1+x[0]))).squeeze()

        # subspace
        tmp = 2 * np.sum((x[0]**2)/(1+x[0]))
        tmp = tmp * np.real(np.trace(u[1].conj().T@v[1]))
        res = np.real(res + tmp)

        return res

    def norm(self, x, u):
        return np.sqrt(self.inner(x, u, u))

    def proj(self, x, G):
        return self._prod_manifold.proj(x, G)

    # def egrad2rgrad(self, x, u):
    #     ...
    #     return _ProductTangentVector(grad)

    # def ehess2rhess(self, x, egrad, ehess, u):

    def _retr(self, x, u):
        res = list()

        # textures
        tmp = x[0] + u[0] + (1/2)*(x[0]**-1)*(u[0]**2)
        res.append(tmp)

        # subspace
        u, _, vh = la.svd(x[1] + u[1], full_matrices=False)
        res.append(u@vh)

        return res

    def _retract_on_manifold(retr_fct):
        # In the case the textures are too small,
        # it reduces the step size until the retracted
        # textures are big enough.
        def f(self, x, u):
            i = 0
            r = retr_fct(self, x, u)
            while (i < ITER_MAX) and (np.sum(r[0] < EPS) > 0):
                u = 1/2 * u
                r = retr_fct(self, x, u)
                i += 1
            if i == ITER_MAX:
                r = x
            return r
        return f

    retr = _retract_on_manifold(_retr)

    def transp(self, x1, x2, d):
        return self.proj(x2, d)
