import autograd
import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as rnd
import pymanopt
from pymanopt.manifolds import ComplexGrassmann
from pymanopt.manifolds.product import _ProductTangentVector
from pymanopt import Problem
from pymanopt.solvers import\
        ConjugateGradient,\
        SteepestDescent
from pymanopt.solvers.linesearch import\
        LineSearchBackTracking
import warnings

from .base import Feature, make_feature_prototype, Product
from .covariance import compute_scm
from ..manifolds import ComplexRobustSubspaceIG, StrictlyPositiveVectors
from ..solvers import StochasticGradientDescent


EPS = 1e-10

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


def wrapper_normalization_sigma(estimation_fct):
    """ A wrapper to normalize X by sigma
    (sqrt of p-k last eigenvalues of SCM).
        Inputs:
            * X = a matrix of size (p, N)
            with each observation along column dimension
            * k = dimension of the subspace
        Outputs:
            * tau
            * U = orthogonal basis of subspace"""
    def estimation(X, k, *args, **kwargs):
        p, N = X.shape
        # normalize vectors by sigma
        eigv = la.eigvalsh(compute_scm(X))[::-1]
        eigv = eigv[k:min(p, N)]
        sigma_2 = np.mean(eigv)
        X = X / np.sqrt(sigma_2)

        res = estimation_fct(X, k, *args, **kwargs)
        tau = res[0]
        U = res[1]

        tau = sigma_2 * tau

        return tau, U, res[2]
    return estimation


def _initialize_estimator(X, k, initialize_textures=True):
    p = X.shape[0]

    # subspace
    if p <= 1e3:
        U = compute_subspace_SCM(X, k)
    else:
        U = np.eye(p)[:, :k]

    # textures
    if initialize_textures:
        proj_X = np.conjugate(U).T @ X
        tmp = la.norm(proj_X, axis=0)**2
        tmp = (1/k) * np.mean(tmp) - 1
        if tmp < EPS:
            tmp = EPS
        tau = tmp*np.ones((X.shape[1], 1))
        return tau, U
    else:
        return U


def _cost(tau, U, X):
    N = X.shape[1]
    k = U.shape[1]

    res = k*np.sum(np.log(1+tau))

    A = np.conj(U).T@X
    tmp = np.einsum('ij,ji->i', np.conj(A).T, A).reshape((N, 1))
    tmp = np.sum((tau/(1+tau)) * tmp)

    res = res - tmp
    res = (1/N)*np.real(res)

    return res


def _egrad(tau, U, X):
    N = X.shape[1]
    k = U.shape[1]
    tau = np.real(tau)
    grad_U = -(2/N)*X@(tau/(1+tau)*X.conj().T)@U
    A = np.conj(U).T@X
    grad_tau = np.einsum('ij,ji->i', (1/(1+tau))*np.conj(A).T, A)
    grad_tau = -(1/N)*np.real((grad_tau[:, np.newaxis] - k)*(1/(1+tau)))
    grad = (grad_tau, grad_U)
    return grad


def create_cost_egrad(backend, X):
    if backend == "Autograd":
        @pymanopt.function.Callable
        def cost(tau, U):
            return _cost(tau, U, X)

        @pymanopt.function.Callable
        def egrad(tau, U):
            res = autograd.grad(cost, argnum=[0, 1])(tau, U)
            res = tuple([np.conjugate(r) for r in res])
            return res

    elif backend == "Callable":
        @pymanopt.function.Callable
        def cost(tau, U):
            return _cost(tau, U, X)

        @pymanopt.function.Callable
        def egrad(tau, U):
            return _egrad(tau, U, X)
    else:
        raise ValueError("Unsupported backend '{:s}'".format(backend))

    return cost, egrad


def _natural_grad(tau, U, X):
    N, k = tau.shape[0], U.shape[1]

    proj_X = np.conjugate(U).T @ X

    # textures
    grad_tau = 1 + tau
    grad_tau = grad_tau - (1/k) * np.einsum(
        'ij,ji->i', np.conjugate(proj_X).T, proj_X).reshape((-1, 1))
    grad_tau = (1/N) * np.real(grad_tau)

    # subspace
    tmp_tau = (tau/(1+tau)).reshape((1, -1))
    grad_U = (tmp_tau * (X - U@proj_X)) @ np.conjugate(proj_X).T
    grad_U = - (1/np.sum((tau**2)/(1+tau))) * grad_U
    grad_U = (1/N) * grad_U

    return _ProductTangentVector((grad_tau, grad_U))


def create_cost_natural_grad(backend, X):
    if backend == "Callable":
        @pymanopt.function.Callable
        def cost(tau, U):
            return _cost(tau, U, X)

        @pymanopt.function.Callable
        def grad(tau, U):
            return _natural_grad(tau, U, X)
    else:
        raise ValueError("Unsupported backend '{:s}'".format(backend))

    return cost, grad


def create_stochastic_cost_natural_grad(backend, X, batch_size):
    if backend == "Callable":
        @pymanopt.function.Callable
        def cost(tau, U):
            return _cost(tau, U, X)

        @pymanopt.function.Callable
        def grad(tau, U):
            batch_idx = rnd.randint(X.shape[1], size=batch_size)
            X_batch = X[:, batch_idx]
            tau_batch = tau[batch_idx, ...]
            grad = _natural_grad(tau_batch, U, X_batch)
            grad_tau = np.zeros_like(tau)
            grad_tau[batch_idx] = grad[0]
            grad[0] = grad_tau
            return grad
    else:
        raise ValueError("Unsupported backend '{:s}'".format(backend))

    return cost, grad


def estimate_tau_UUH_RO(
    X,
    k,
    init=None,
    information_geometry=False,
    min_step_size=1e-14,
    iter_max=int(1e3),
    time_max=np.inf,
    solver='conjugate',
    autodiff=False,
    batch_size=1000,
    verbosity=0
):
    """ A function that estimates parameters of a 'tau UUH' model.
        Inputs:
            * X = a matrix of size (p, N)
            with each observation along column dimension
            * k = dimension of the subspace
            * init = point on manifold to initliase estimation
            * information_geometry = to use or not the natural gradient
            * min_step_size = minimum step size at each iteration
            * iter_max = maximum number of iterations
            * time_max = maximum time
            * solver = 'steepest' or 'conjugate' or 'SGD'
            * autodiff = use or not automatic differenciation
            * batch_size = only used when solver is 'SGD'
            * verbosity = verbosity of optimizer
        Outputs:
            * tau
            * U = orthogonal basis of subspace
            * log"""

    p, N = X.shape

    # Initialisation
    if init is None:
        init = _initialize_estimator(X, k)

    if autodiff:
        backend = 'Autograd'
    else:
        backend = 'Callable'

    if solver == 'SGD':
        batch_size = np.min([X.shape[1], batch_size])
        if information_geometry:
            cost, grad = create_stochastic_cost_natural_grad(
                backend, X, batch_size=batch_size)
            manifold = ComplexRobustSubspaceIG(N, p, rank=k)
        else:
            error_str = 'SGD with simple gardient (not natural one)'
            error_str += ' is not implemented...'
            raise ValueError(error_str)
        problem = Problem(
            manifold=manifold, cost=cost, grad=grad, verbosity=verbosity)
        solver = StochasticGradientDescent(
            checkperiod=100,
            stepsize_init=10, stepsize_type='decay',
            stepsize__lambda=0.005, stepsize_decaysteps=None,
            maxtime=time_max, maxiter=iter_max,
            minstepsize=min_step_size,
            maxcostevals=np.inf, logverbosity=2)

    elif solver in ['steepest', 'conjugate']:
        if information_geometry:
            cost, grad = create_cost_natural_grad(backend, X)
            manifold = ComplexRobustSubspaceIG(N, p, rank=k)
            problem = Problem(
                manifold=manifold, cost=cost, grad=grad, verbosity=verbosity)
            linesearch = LineSearchBackTracking(
                contraction_factor=.95,
                optimism=100,
                suff_decr=1e-16,
                maxiter=200,
                initial_stepsize=10
            )
        else:
            cost, egrad = create_cost_egrad(backend, X)
            manifold = Product([StrictlyPositiveVectors(N),
                                ComplexGrassmann(p, k)])

            problem = Problem(manifold=manifold,
                              cost=cost, egrad=egrad, verbosity=0)
            linesearch = LineSearchBackTracking()
        if solver == 'steepest':
            solver = SteepestDescent
        elif solver == 'conjugate':
            solver = ConjugateGradient
        solver = solver(linesearch=linesearch,
                        maxtime=time_max, maxiter=iter_max,
                        minstepsize=min_step_size,
                        maxcostevals=np.inf, logverbosity=2)

    else:
        error_str = 'Solvers available: steepest, conjugate and SGD.'
        raise ValueError(error_str)

    Xopt, log = solver.solve(problem, x=init)

    return Xopt[0], Xopt[1], log


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

    # Initialisation
    delta = np.inf  # Distance between two iterations
    U = _initialize_estimator(X, k, initialize_textures=False)
    iteration = 0

    # store likelihood values
    cost, _ = create_cost_egrad('Callable', X)
    likelihood_values = list()

    while (delta > tol) and (iteration < iter_max):
        # compute tau
        X_projected = U.conj().T@X
        tau_new = np.einsum('ij,ji->i', X_projected.conj().T, X_projected)
        tau_new = (1/k)*np.real(tau_new)-1
        tau_new[tau_new <= EPS] = EPS

        # store likelihood
        likelihood_values.append(cost(tau_new.reshape((-1, 1)), U))

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

    log = dict()
    log['iterations'] = dict()
    log['iterations']['f(x)'] = likelihood_values

    if iteration == iter_max:
        warnings.warn('Estimation algorithm did not converge')

    return tau, U, log


# CLASSES


@make_feature_prototype
def subspace_SCM(
    k,
    assume_centered=True,
    min_grad_norm_mean=1e-6,
    p=None,
    N=None
):
    if assume_centered:
        name = 'subspace_SCM'
    else:
        name = 'subspace_centered_SCM'
    name += '_k_' + str(k)

    M = ComplexGrassmann
    args_M = {'sizes': (p, k)}

    def _subspace_SCM(X):
        return compute_subspace_SCM(X, k, assume_centered=assume_centered)

    return Feature(name, _subspace_SCM, M, args_M,
                   min_grad_norm_mean=min_grad_norm_mean)


@make_feature_prototype
def subspace_tau_UUH(
    k,
    estimate_sigma=False,
    min_grad_norm_mean=1e-6,
    p=None,
    N=None
):
    name = 'subspace_tau_UUH'
    name += '_k_' + str(k)

    M = ComplexGrassmann
    args_M = {'sizes': (p, k)}

    def _subspace_tau_UUH(X):
        if estimate_sigma:
            _, U, _ = wrapper_normalization_sigma(estimate_tau_UUH)(X, k)
        else:
            _, U, _ = estimate_tau_UUH(X, k)
        return U

    return Feature(name, _subspace_tau_UUH, M, args_M,
                   min_grad_norm_mean=min_grad_norm_mean)


@make_feature_prototype
def subspace_tau_UUH_RO(
    k,
    estimate_sigma=False,
    information_geometry=False,
    min_step_size=1e-14,
    iter_max=int(1e3),
    time_max=np.inf,
    solver='conjugate',
    autodiff=False,
    batch_size=1000,
    verbosity=0,
    min_grad_norm_mean=1e-6,
    p=None,
    N=None
):
    name = 'subspace_tau_UUH'
    if information_geometry:
        name += '_IG'
    else:
        name += '_RO'
    name += '_k_' + str(k)

    M = ComplexGrassmann
    args_M = {'sizes': (p, k)}

    def _subspace_tau_UUH_RO(X):
        est_fct = estimate_tau_UUH_RO
        if estimate_sigma:
            est_fct = wrapper_normalization_sigma(est_fct)
        _, U, _ = est_fct(
            X, k,
            information_geometry=information_geometry,
            min_step_size=min_step_size,
            iter_max=iter_max,
            time_max=time_max,
            solver=solver,
            autodiff=autodiff,
            batch_size=batch_size,
            verbosity=verbosity
        )
        return U

    return Feature(name, _subspace_tau_UUH_RO, M, args_M,
                   min_grad_norm_mean=min_grad_norm_mean)


@make_feature_prototype
def tau_UUH(
    k,
    estimate_sigma=False,
    weights=None,
    min_grad_norm_mean=1e-6,
    p=None,
    N=None
):
    if weights is None:
        name = 'tau_UUH'
    else:
        name = 'tau_' + str(round(weights[0], 5)) +\
               '_UUH_' + str(round(weights[1], 5))
    name += '_k_' + str(k)

    M = (StrictlyPositiveVectors, ComplexGrassmann)
    if weights is None:
        weights = (1/N, 1/k)
    args_M = {
        'sizes': (N, (p, k)),
        'weights': weights
    }

    def _estimate_tau_UUH(X):
        if estimate_sigma:
            tau, U, _ = wrapper_normalization_sigma(estimate_tau_UUH)(X, k)
        else:
            tau, U, _ = estimate_tau_UUH(X, k)
        return tau, U

    return Feature(name, _estimate_tau_UUH, M, args_M,
                   min_grad_norm_mean=min_grad_norm_mean)


@make_feature_prototype
def tau_UUH_RO(
    k,
    estimate_sigma=False,
    information_geometry=False,
    min_step_size=1e-14,
    iter_max=int(1e3),
    time_max=np.inf,
    solver='conjugate',
    autodiff=False,
    batch_size=1000,
    weights=None,
    verbosity=0,
    min_grad_norm_mean=1e-6,
    p=None,
    N=None
):
    if weights is None:
        name = 'tau_UUH'
    else:
        name = 'tau_' + str(round(weights[0], 5)) +\
               '_UUH_' + str(round(weights[1], 5))
    if information_geometry:
        name += '_IG'
    else:
        name += '_RO'
    name += '_k_' + str(k)

    M = (StrictlyPositiveVectors, ComplexGrassmann)
    if weights is None:
        weights = (1/N, 1/k)
    args_M = {
        'sizes': (N, (p, k)),
        'weights': weights
    }

    def _estimate_tau_UUH_RO(X):
        est_fct = estimate_tau_UUH_RO
        if estimate_sigma:
            est_fct = wrapper_normalization_sigma(est_fct)
        tau, U, _ = est_fct(
            X, k,
            information_geometry=information_geometry,
            min_step_size=min_step_size,
            iter_max=iter_max,
            time_max=time_max,
            solver=solver,
            autodiff=autodiff,
            batch_size=batch_size,
            verbosity=verbosity
        )
        return tau, U

    return Feature(name, _estimate_tau_UUH_RO, M, args_M,
                   min_grad_norm_mean=min_grad_norm_mean)
