import autograd
import autograd.numpy as np
import autograd.numpy.linalg as la
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import\
        ComplexEuclidean,\
        StrictlyPositiveVectors,\
        SpecialHermitianPositiveDefinite
from pymanopt.solvers import ConjugateGradient, SteepestDescent, TrustRegions
import warnings

from .base import Feature, make_feature_prototype, Product
from ..manifolds import ComplexCompoundGaussianIG


# Gaussian estimation


def Gaussian_estimation(X):
    """ Function that computes Gaussian estimators: sample mean and SCM
        Inputs:
            * X = a matrix of size p*N
        Outputs:
            * mu = estimate of location
            * sigma = estimate of covariance"""
    p, N = X.shape
    mu = np.mean(X, axis=1).reshape((-1, 1))
    sigma = (1/N) * (X-mu) @ (X-mu).conj().T
    c = np.real(la.det(sigma)**(1/p))
    tau = c * np.ones((N, 1))
    sigma = sigma / c
    return mu, sigma, tau


# ESTIMATION


def tyler_estimator_location_covariance_normalized_det(
    X,
    init=None,
    tol=0.001,
    iter_max=100
):
    """ A function that computes the Tyler Fixed Point Estimator for
        location and covariance estimation.
        Inputs:
            * X = a matrix of size p*N
            * init = point on manifold to initialise estimation
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * mu = estimate of location
            * sigma = estimate of covariance
            * delta = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    p, N = X.shape
    if init is None:
        mu = np.mean(X, axis=1).reshape((-1, 1))
        sigma = (1/N)*(X-mu)@(X-mu).conj().T
        sigma = sigma/(la.det(sigma)**(1/p))
    else:
        mu, _, sigma = init

    mu = mu.reshape((-1, 1))

    delta = np.inf  # Distance between two iterations
    iteration = 0

    while (delta > tol) and (iteration < iter_max):
        # compute mu (location)
        tau = np.sqrt(np.real(np.einsum('ij,ji->i',
                                        np.conjugate(X-mu).T@la.inv(sigma),
                                        X-mu)))
        mu = (1/np.sum(1/tau)) * np.sum(X/tau, axis=1).reshape((-1, 1))

        # compute sigma
        tau = np.sqrt(np.real(np.einsum('ij,ji->i',
                                        np.conjugate(X-mu).T@la.inv(sigma),
                                        X-mu)))
        X_bis = (X-mu) / tau
        sigma_new = (p/N) * X_bis@X_bis.conj().T

        # condition for stopping
        delta = la.norm(sigma_new - sigma, 'fro') / la.norm(sigma, 'fro')
        iteration = iteration + 1

        # updating
        sigma = sigma_new

    # recomputing tau
    tau = np.real(np.einsum('ij,ji->i',
                            np.conjugate(X-mu).T@la.inv(sigma),
                            X-mu))

    # imposing det constraint: det(sigma_new) = 1
    c = np.real(la.det(sigma)**(1/p))
    sigma = sigma/c
    tau = c*tau

    if iteration == iter_max:
        warnings.warn('Estimation algorithm did not converge')

    mu = mu.reshape((-1, 1))
    tau = tau.reshape((-1, 1))

    return mu, sigma, tau, delta, iteration


def create_cost_egrad_ehess_location_covariance_texture(X, autodiff=False):
    @pymanopt.function.Callable
    def cost(mu, sigma, tau):
        p, N = X.shape

        # compute quadratic form
        X_bis = X-mu
        Q = np.real(np.einsum('ij,ji->i',
                              np.conjugate(X_bis).T@la.inv(sigma),
                              X_bis))

        tau = np.squeeze(tau)
        L = p*np.log(tau) + Q/tau
        L = np.real(np.sum(L))

        return L

    @pymanopt.function.Callable
    def egrad(mu, sigma, tau):
        p, N = X.shape
        sigma_inv = la.inv(sigma)

        # grad mu
        grad_mu = (X-mu)/tau.T
        grad_mu = np.sum(grad_mu, axis=1, keepdims=True)
        grad_mu = -2*sigma_inv@grad_mu

        # grad sigma
        X_bis = (X-mu) / np.sqrt(tau).T
        grad_sigma = X_bis@X_bis.conj().T
        grad_sigma = -sigma_inv@grad_sigma@sigma_inv

        # grad tau
        X_bis = X-mu
        a = np.real(np.einsum('ij,ji->i',
                              np.conjugate(X_bis).T@la.inv(sigma),
                              X_bis))
        a = a.reshape((-1, 1))
        grad_tau = np.real((p*tau-a) * (tau**(-2)))

        # grad
        grad = (grad_mu, grad_sigma, grad_tau)

        return grad

    @pymanopt.function.Callable
    def ehess(mu, sigma, tau, xi_mu, xi_sigma, xi_tau):
        p, N = X.shape
        sigma_inv = la.inv(sigma)
        X_bis = X-mu

        # hess mu
        tmp0 = X_bis/tau.T
        tmp1 = np.sum((tmp0*xi_tau.T)/tau.T, axis=1, keepdims=True)
        tmp0 = np.sum(tmp0, axis=1, keepdims=True)
        tmp2 = np.sum(1/tau.T)
        hess_mu = 2*sigma_inv @ (xi_sigma@sigma_inv@tmp0 + tmp2*xi_mu + tmp1)

        # hess sigma
        def herm(s):
            return (s + np.conjugate(s).T)/2
        tmp0 = X_bis / np.sqrt(tau).T
        tmp0 = 2*herm(xi_sigma@sigma_inv@(tmp0@np.conjugate(tmp0).T))
        tmp1 = np.tile(np.conjugate(xi_mu).T, (X_bis.shape[1], 1))
        tmp1 = 2*herm(X_bis / tau.T @ tmp1)
        tmp2 = (X_bis / tau.T)
        tmp2 = (tmp2*xi_tau.T) @ np.conjugate(tmp2).T
        hess_sigma = sigma_inv@(tmp0 + tmp1 + tmp2)@sigma_inv

        # hess tau
        tmp0 = sigma_inv @ X_bis
        tmp1 = 2*np.real(np.einsum('ij,ji->i', np.conjugate(xi_mu).T, tmp0))
        tmp1 = tmp1.reshape((-1, 1))
        tmp2 = np.real(np.einsum(
            'ij,ji->i', np.conjugate(tmp0).T@xi_sigma, tmp0))
        tmp2 = tmp2.reshape((-1, 1))
        a = np.real(np.einsum('ij,ji->i', np.conjugate(X_bis).T, tmp0))
        a = a.reshape((-1, 1))
        hess_tau = p*xi_tau + tmp1 + tmp2 + 2*(a - p*tau)*xi_tau*(tau**(-1))
        hess_tau = hess_tau * (tau**(-2))

        # hess
        hess = (hess_mu, hess_sigma, hess_tau)

        return hess

    @pymanopt.function.Callable
    def auto_egrad(mu, sigma, tau):
        res = autograd.grad(cost, argnum=[0, 1, 2])(mu, sigma, tau)
        res = tuple(np.conjugate(res))
        return res

    @pymanopt.function.Callable
    def auto_ehess(mu, sigma, tau, xi_mu, xi_sigma, xi_tau):
        def directional_derivative(mu, sigma, tau, xi_mu, xi_sigma, xi_tau):
            tmp = autograd.grad(cost, argnum=[0, 1, 2])(mu, sigma, tau)
            gradients = [np.conjugate(g) for g in tmp]
            directions = (xi_mu, xi_sigma, xi_tau)
            res = np.sum([np.tensordot(
                gradient,
                np.conjugate(vector),
                axes=vector.ndim
            )
                          for gradient, vector in zip(gradients, directions)])
            return np.real(res)
        ehess = autograd.grad(
            directional_derivative,
            argnum=[0, 1, 2]
        )(mu, sigma, tau, xi_mu, xi_sigma, xi_tau)
        ehess = tuple(np.conjugate(ehess))
        return ehess

    if autodiff:
        return cost, auto_egrad, auto_ehess
    else:
        return cost, egrad, ehess


def estimate_location_covariance_texture(
    X,
    init=None,
    information_geometry=False,
    tol=1e-3,
    iter_max=3*int(1e4),
    time_max=np.inf,
    autodiff=False,
    solver='conjugate'
):
    """ A function that estimates parameters of a compound Gaussian distribution.
        Inputs:
            * X = a matrix of size p*N
            * init = point on manifold to initliase estimation
            * tol = minimum norm of gradient
            * iter_max = maximum number of iterations
            * time_max = maximum time in seconds
            * autodiff = use or not autodiff
            * solver = steepest or conjugate
        Outputs:
            * mu = estimate of location
            * tau = estimate of tau
            * sigma = estimate of covariance """

    # The estimation is done using Riemannian geometry.
    p, N = X.shape

    # Initialisation
    if init is None:
        mu = np.mean(X, axis=1).reshape((-1, 1))
        sigma = (1/N)*(X-mu)@(X-mu).conj().T
        tau = np.real(la.det(sigma)**(1/p))*np.ones((1, N))
        sigma = sigma/(la.det(sigma)**(1/p))
        init = [mu, sigma, tau]

    init[0] = init[0].reshape((-1, 1))
    init[2] = init[2].reshape((-1, 1))

    # cost, egrad, ehess
    cost, egrad, ehess = create_cost_egrad_ehess_location_covariance_texture(
        X,
        autodiff
    )

    # solver
    solver_str = solver
    if solver == 'steepest':
        solver = SteepestDescent
    elif solver == 'conjugate':
        solver = ConjugateGradient
    elif solver == 'trust-regions':
        solver = TrustRegions
    else:
        s = 'Solvers available: steepest, conjugate, trust-regions.'
        raise ValueError(s)
    solver = solver(
        maxtime=time_max,
        maxiter=iter_max,
        mingradnorm=tol,
        minstepsize=0,
        maxcostevals=np.inf,
        logverbosity=2
    )

    # manifold
    if information_geometry:
        manifold = ComplexCompoundGaussianIG(p, N)
    else:
        SHPD = SpecialHermitianPositiveDefinite(p)
        if solver_str == 'trust-regions':
            # exp seems more stable than retr when using trust-regions
            SHPD.retr = SHPD.exp
        manifold = Product([
            ComplexEuclidean(p, 1),
            SHPD,
            StrictlyPositiveVectors(N)])

    problem = Problem(
        manifold=manifold,
        cost=cost,
        egrad=egrad,
        ehess=ehess,
        verbosity=0
    )
    Xopt, log = solver.solve(problem, x=init)

    return Xopt[0], Xopt[1], Xopt[2], log


# CLASSES

@make_feature_prototype
def location_covariance_texture_Gaussian(
    weights=(1, 1, 1),
    p=None,
    N=None,
    **kwargs
):
    name = 'location_covariance_texture_Gaussian'
    M = (ComplexEuclidean,
         SpecialHermitianPositiveDefinite,
         StrictlyPositiveVectors)
    args_M = {
        'sizes': (p, p, N),
        'weights': weights
    }

    return Feature(name, Gaussian_estimation, M, args_M)


@make_feature_prototype
def location_covariance_texture_Tyler(
    weights=(1, 1, 1),
    p=None,
    N=None,
    **kwargs
):
    name = 'location_covariance_texture_Tyler'
    M = (ComplexEuclidean,
         SpecialHermitianPositiveDefinite,
         StrictlyPositiveVectors)
    args_M = {
        'sizes': (p, p, N),
        'weights': weights
    }

    def _estimation(X):
        mu, sigma, tau, _, _ =\
                tyler_estimator_location_covariance_normalized_det(X)
        return mu, sigma, tau

    return Feature(name, _estimation, M, args_M)


@make_feature_prototype
def location_covariance_texture(
    iter_max=3*int(1e4),
    information_geometry=False,
    solver='trust-regions',
    weights=(1, 1, 1),
    p=None,
    N=None,
    **kwargs
):
    name = 'location_covariance_texture'
    M = (ComplexEuclidean,
         SpecialHermitianPositiveDefinite,
         StrictlyPositiveVectors)
    args_M = {
        'sizes': (p, p, N),
        'weights': weights
    }

    def _estimation(X):
        mu, sigma, tau, _ = estimate_location_covariance_texture(
            X,
            information_geometry=information_geometry,
            iter_max=iter_max,
            solver=solver
        )
        return mu, sigma, tau

    return Feature(name, _estimation, M, args_M)
