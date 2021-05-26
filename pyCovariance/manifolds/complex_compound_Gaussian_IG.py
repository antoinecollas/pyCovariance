import autograd.numpy as np
from autograd.numpy import linalg as la

from pyCovariance.features.base import Product
from pymanopt.manifolds import\
        ComplexEuclidean,\
        SpecialHermitianPositiveDefinite,\
        StrictlyPositiveVectors
from pymanopt.manifolds.manifold import Manifold
from pymanopt.manifolds.product import _ProductTangentVector
from pymanopt.tools.multi import multiherm, multiprod


class ComplexCompoundGaussianIG(Manifold):
    """Information geometry of complex Compound Gaussian distributions.
    3 parameters: location, scatter matrix, textures.
    """
    def __init__(self, p, n, k=1):
        # p: dimension of data/location
        self._p = p
        # n: number of data/ dimension of textures
        self._n = n
        # k: optimize several distributions at the same time
        # k>1: not implemented for the moment
        self._k = k
        if k > 1:
            raise NotImplementedError

        if k == 1:
            name = ('Information geometry of \
                    complex Compound Gaussian distributions \
                    ({} location vectors,\
                    {} x {} scatter matrices,\
                    {} textures vectors').format(p, p, p, n)
        else:
            name = ('Product manifold of \
                    information geometry of \
                    complex Compound Gaussian distributions \
                    ({} location vectors, \
                    {} x {} scatter matrices, \
                    {} textures vectors').format(p, p, p, n)
        dimension_location = int(2 * k * p)
        dimension_scatter = int(k * p * p)
        dimension_textures = int(k * n)
        dimension = dimension_location + dimension_scatter + dimension_textures
        prod_manifold = Product([
            ComplexEuclidean(p, k),
            SpecialHermitianPositiveDefinite(p, k),
            StrictlyPositiveVectors(n, k)])
        self._prod_manifold = prod_manifold
        super().__init__(name, dimension, point_layout=3)

    def rand(self):
        return self._prod_manifold.rand()

    def randvec(self, x):
        u = self._prod_manifold.randvec(x)
        u = u * (1/self.norm(x, u))
        return u

    def zerovec(self, x):
        return self._prod_manifold.zerovec(x)

    def inner(self, x, u, v):
        man = self._prod_manifold._manifolds
        n, p = self._n, self._p

        # location
        c = la.cholesky(x[1])
        c_inv = la.inv(c)
        res = man[0].inner(
            x[0],
            multiprod(c_inv, u[0]),
            multiprod(c_inv, v[0])
        )
        res = 2 * np.sum(1/x[2], axis=0) * res

        # scatter
        res = res + n * man[1].inner(x[1], u[1], v[1])

        # textures
        res = res + p * man[2].inner(x[2], u[2], v[2])

        if self._k == 1:
            return res.reshape(1)[0]

    def norm(self, x, u):
        man = self._prod_manifold._manifolds
        n, p = self._n, self._p

        # location
        c = la.cholesky(x[1])
        c_inv = la.inv(c)
        res = man[0].norm(x[0], multiprod(c_inv, u[0]))**2
        res = 2 * np.sum(1/x[2], axis=0) * res

        # scatter
        res = res + n * man[1].norm(x[1], u[1])**2

        # textures
        res = res + p * man[2].norm(x[2], u[2])**2

        res = np.sqrt(res)

        if self._k == 1:
            return res.reshape(1)[0]

    def proj(self, x, G):
        return self._prod_manifold.proj(x, G)

    def egrad2rgrad(self, x, u):
        man = self._prod_manifold._manifolds
        n, p = self._n, self._p
        grad = list()

        # location
        grad.append((1/(2*np.sum(1/x[2])))*multiprod(x[1], u[0]))

        # scatter
        grad.append((1/n) * man[1].egrad2rgrad(x[1], u[1]))

        # textures
        grad.append((1/p) * man[2].egrad2rgrad(x[2], u[2]))

        return _ProductTangentVector(grad)

    # def ehess2rhess(self, x, egrad, ehess, u):
    #     egrad = multiherm(egrad)
    #     hess = multiprod(multiprod(x, multiherm(ehess)), x)
    #     hess += multiherm(multiprod(multiprod(u, egrad), x))
    #     return hess

    def retr(self, x, u):
        n, p = self._n, self._p
        sigma_inv = la.inv(x[1])
        r = list()

        # location
        c = np.sum(u[2]*(x[2]**(-2))) / np.sum(1/x[2])
        c = c * np.eye(p)
        c = c + multiprod(u[1], sigma_inv)
        c = multiprod(c, u[0])
        r.append(x[0] + u[0] + (1/2) * c)

        # scatter
        c = multiprod(multiprod(u[1], sigma_inv), u[1])
        c = c - (1/n)*np.sum(1/x[2]) * multiprod(u[0], u[0].conj().T)
        sigma = x[1] + u[1] + (1/2) * c
        sigma = sigma / np.real(la.det(sigma)**(1/p))
        sigma = multiherm(sigma)
        r.append(sigma)

        # textures
        c = (u[2]**2) * (1/x[2])
        tmp = multiprod(multiprod(u[0].conj().T, sigma_inv), u[0])
        tmp = np.real(tmp)
        c = c - (1/p) * tmp * np.ones((n, 1))
        t = x[2] + u[2] + (1/2) * c
        t[t < 1e-12] = 1e-12  # numerically necessary
        r.append(t)

        return r

    # def transp(self, x1, x2, d):
    #     E = multihconj(la.solve(multihconj(x1), multihconj(x2)))
    #     if self._k == 1:
    #         E = sqrtm(E)
    #     else:
    #         for i in range(len(E)):
    #             E[i, :, :] = sqrtm(E[i, :, :])
    #     transp_d = multiprod(multiprod(E, d), multihconj(E))
    #     return transp_d
