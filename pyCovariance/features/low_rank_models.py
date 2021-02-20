import autograd
import autograd.numpy as np
import autograd.numpy.linalg as la
import pymanopt
from pymanopt.manifolds import\
        ComplexGrassmann,\
        StrictlyPositiveVectors
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient
import warnings

from .base import Feature, make_feature_prototype, Product
from .covariance import compute_scm


# SUBSPACE SCM ESTIMATION


def compute_subspace_SCM(X, k, assume_centered=True):
    """ A function that estimates the subspace of
    the sample covariance matrix (SCM).
        Inputs:
            * X = a matrix of size (p, N)
            with each observation along column dimension
            * k = dimension of the subspace
            * assume_centered = bool.
            If False, data are centered with empirical mean.
        Outputs:
            * U = orthogonal basis of subspace"""
    if not assume_centered:
        mean = np.mean(X, axis=1, keepdims=True)
        X = X - mean
    U, _, _ = la.svd(X, full_matrices=False)
    U = U[:, :k]
    return U


# tau UUH ESTIMATION


def _cost(U, tau, X):
    n = X.shape[1]
    k = U.shape[1]
    res = np.sum(np.einsum('ij,ji->i', np.conj(X).T, X))
    A = np.conj(U).T@X
    res = res - np.sum(np.einsum('ij,ji->i', (tau/(1+tau))*np.conj(A).T, A))
    res = res + k*np.sum(np.log(1+tau))
    res = (1/n)*np.real(res)
    return res


def _egrad(U, tau, X):
    n = X.shape[1]
    k = U.shape[1]
    tau = np.real(tau)
    grad_U = -(2/n)*X@(tau/(1+tau)*X.conj().T)@U
    A = np.conj(U).T@X
    grad_tau = np.einsum('ij,ji->i', (1/(1+tau))*np.conj(A).T, A)
    grad_tau = -(1/n)*np.real((grad_tau[:, np.newaxis] - k)*(1/(1+tau)))
    grad = (grad_U, grad_tau)
    return grad


def create_cost_egrad(backend, X):
    if backend == "Autograd":
        @pymanopt.function.Callable
        def cost(U, tau):
            return _cost(U, tau, X)

        @pymanopt.function.Callable
        def egrad(U, tau):
            return tuple(np.conj(autograd.grad(cost, argnum=[0, 1])(U, tau)))

    elif backend == "Callable":
        @pymanopt.function.Callable
        def cost(U, tau):
            return _cost(U, tau, X)

        @pymanopt.function.Callable
        def egrad(U, tau):
            return _egrad(U, tau, X)
    else:
        raise ValueError("Unsupported backend '{:s}'".format(backend))

    return cost, egrad


def estimate_tau_UUH_RGD(
    X,
    k,
    init=None,
    iter_max=int(1e3),
    autodiff=False
):
    """ A function that estimates parameters of a 'tau UUH' model.
        Inputs:
            * X = a matrix of size (p, N)
            with each observation along column dimension
            * k = dimension of the subspace
            * init = point on manifold to initliase estimation
            * iter_max = maximum number of iterations
            * autodiff = use or not autodiff
        Outputs:
            * U = orthogonal basis of subspace
            * tau """

    p, N = X.shape

    # normalize vectors by sigma
    eigv = la.eigvalsh(compute_scm(X))[::-1]
    eigv = eigv[k:min(p, N)]
    sigma_2 = np.mean(eigv)
    X = X / np.sqrt(sigma_2)

    # Initialisation
    if init is None:
        U = compute_subspace_SCM(X, k)
        X_projected = U.conj().T@X
        tau = np.einsum('ij,ji->i', X_projected.conj().T, X_projected)
        tau = (1/k)*np.real(tau)-1
        tau[tau <= 1e-10] = 1e-10
        tau = tau.reshape((-1, 1))
        init = (U, tau)

    if autodiff:
        backend = 'Autograd'
    else:
        backend = 'Callable'

    cost, egrad = create_cost_egrad(backend, X)
    manifold = Product([ComplexGrassmann(p, k),
                        StrictlyPositiveVectors(N)])

    problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=0)

    solver = ConjugateGradient(
        maxiter=iter_max
    )
    parameters = solver.solve(problem, x=init)
    tau, U = parameters[1], parameters[0]

    tau = sigma_2 * tau

    return tau, U


def estimate_tau_UUH(X, k, tol=0.001, iter_max=1000):
    """ A function that estimates parameters of a 'tau UUH' model.
        Inputs:
            * X = a matrix of size (p, N)
            with each observation along column dimension
            * k = dimension of subspace
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * U = orthogonal basis of subspace
            * tau """
    p, N = X.shape

    # normalize vectors by sigma
    eigv = la.eigvalsh(compute_scm(X))[::-1]
    eigv = eigv[k:min(p, N)]
    sigma_2 = np.mean(eigv)
    X = X / np.sqrt(sigma_2)

    # Initialisation
    delta = np.inf  # Distance between two iterations
    U = compute_subspace_SCM(X, k)
    iteration = 0

    while (delta > tol) and (iteration < iter_max):
        # compute tau
        X_projected = U.conj().T@X
        tau_new = np.einsum('ij,ji->i', X_projected.conj().T, X_projected)
        tau_new = (1/k)*np.real(tau_new)-1
        tau_new[tau_new <= 1e-10] = 1e-10

        # compute U
        pi = (tau_new/(1+tau_new))*X@X.conj().T

        U_new, _, _ = la.svd(pi)
        U_new = U_new[:, :k]

        # condition for stopping
        delta = la.norm(U_new@U_new.conj().T - U@U.conj().T)
        delta = delta / la.norm(U@U.conj().T)
        iteration = iteration + 1

        # updating
        tau = tau_new
        U = U_new

    tau = tau.reshape((-1, 1))
    tau = sigma_2 * tau

    if iteration == iter_max:
        warnings.warn('Estimation algorithm did not converge')

    return tau, U


# CLASSES


@make_feature_prototype
def subspace_SCM(k, assume_centered=True, p=None, **kwargs):
    if assume_centered:
        name = 'subspace_SCM'
    else:
        name = 'subspace_centered_SCM'
    name += '_k_' + str(k)

    M = ComplexGrassmann
    args_M = {'sizes': (p, k)}

    def _subspace_SCM(X):
        return compute_subspace_SCM(X, k, assume_centered=assume_centered)

    return Feature(name, _subspace_SCM, M, args_M)


@make_feature_prototype
def subspace_tau_UUH(k, p, **kwargs):
    name = 'subspace_tau_UUH'
    name += '_k_' + str(k)

    M = ComplexGrassmann
    args_M = {'sizes': (p, k)}

    def _subspace_tau_UUH(X):
        _, U = estimate_tau_UUH(X, k)
        return U

    return Feature(name, _subspace_tau_UUH, M, args_M)


@make_feature_prototype
def subspace_tau_UUH_RGD(k, autodiff=False, p=None, **kwargs):
    name = 'subspace_tau_UUH_RGD'
    name += '_k_' + str(k)

    M = ComplexGrassmann
    args_M = {'sizes': (p, k)}

    def _subspace_tau_UUH_RGD(X):
        _, U = estimate_tau_UUH_RGD(X, k, autodiff=autodiff)
        return U

    return Feature(name, _subspace_tau_UUH_RGD, M, args_M)


@make_feature_prototype
def tau_UUH(k, weights=None, p=None, N=None, **kwargs):
    if weights is None:
        name = 'tau_UUH'
    else:
        name = 'tau_' + str(round(weights[0], 4)) +\
               '_UUH_' + str(round(weights[1], 4))
    name += '_k_' + str(k)

    M = (StrictlyPositiveVectors, ComplexGrassmann)
    if weights is None:
        weights = (1/N, 1/k)
    args_M = {
        'sizes': (N, (p, k)),
        'weights': weights
    }

    def _estimate_tau_UUH(X):
        return estimate_tau_UUH(X, k)

    return Feature(name, _estimate_tau_UUH, M, args_M)


@make_feature_prototype
def tau_UUH_RGD(k, weights=None, p=None, N=None, **kwargs):
    if weights is None:
        name = 'tau_UUH_RGD'
    else:
        name = 'tau_' + str(round(weights[0], 4)) +\
               '_UUH_' + str(round(weights[1], 4)) +\
                '_RGD'
    name += '_k_' + str(k)

    M = (StrictlyPositiveVectors, ComplexGrassmann)
    if weights is None:
        weights = (1/N, 1/k)
    args_M = {
        'sizes': (N, (p, k)),
        'weights': weights
    }

    def _estimate_tau_UUH(X):
        return estimate_tau_UUH_RGD(X, k)

    return Feature(name, _estimate_tau_UUH, M, args_M)
