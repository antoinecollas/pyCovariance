import autograd.numpy as np
import pymanopt.manifolds as man
from pymanopt.manifolds.product import _ProductTangentVector


class Product(man.Product):
    """Product manifold with linear combination of metrics."""

    def __init__(self, manifolds, weights=None):
        if weights is None:
            weights = np.ones(len(manifolds))
        self._weights = tuple(weights)
        super().__init__(manifolds)

    @property
    def typicaldist(self):
        raise NotImplementedError

    def inner(self, X, G, H):
        weights = self._weights
        return np.sum([weights[k]*np.squeeze(man.inner(X[k], G[k], H[k]))
                       for k, man in enumerate(self._manifolds)])

    def dist(self, X, Y):
        weights = self._weights
        return np.sqrt(np.sum([weights[k]*np.squeeze(man.dist(X[k], Y[k])**2)
                               for k, man in enumerate(self._manifolds)]))

    def egrad2rgrad(self, X, U):
        weights = self._weights
        return _ProductTangentVector(
            [(1/weights[k])*man.egrad2rgrad(X[k], U[k])
             for k, man in enumerate(self._manifolds)])

    def ehess2rhess(self, X, egrad, ehess, H):
        # Using Koszul formula and R-linearity of affine connections, we get
        # that the Riemannian Hessian of a weighted product manifold
        # is the tuple of the Riemannian Hessians of the different manifolds
        # multiplied by the inverted weights.
        weights = self._weights
        return _ProductTangentVector(
            [(1/weights[k])*man.ehess2rhess(
                X[k], egrad[k], ehess[k], H[k])
             for k, man in enumerate(self._manifolds)])

    def randvec(self, X):
        weights = self._weights
        scale = len(self._manifolds) ** (-1/2)
        return _ProductTangentVector(
            [scale * (1/weights[k]**(-1/2)) * man.randvec(X[k])
             for k, man in enumerate(self._manifolds)])
