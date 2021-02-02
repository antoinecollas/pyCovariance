import autograd
import autograd.numpy as np
import autograd.numpy.linalg as la
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import\
        ComplexEuclidean,\
        StrictlyPositiveVectors,\
        SpecialHermitianPositiveDefinite
from pymanopt.solvers import ConjugateGradient, SteepestDescent
import warnings

from .base import Feature, make_feature_prototype, Product


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
    iter_max=20
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


def create_cost_egrad_location_covariance_texture(X, autodiff=False):
    @pymanopt.function.Callable
    def cost(mu, tau, sigma):
        p, N = X.shape

        # compute quadratic form
        mu = mu.reshape((-1, 1))
        X_bis = X-mu
        Q = np.real(np.einsum('ij,ji->i',
                              np.conjugate(X_bis).T@la.inv(sigma),
                              X_bis))

        tau = np.squeeze(tau)
        L = p*np.log(tau) + Q/tau
        L = np.real(np.sum(L))

        return L

    @pymanopt.function.Callable
    def egrad(mu, tau, sigma):
        p, N = X.shape
        sigma_inv = la.inv(sigma)

        # grad mu
        mu = mu.reshape((-1, 1))
        grad_mu = (X-mu)/tau.T
        grad_mu = np.sum(grad_mu, axis=1, keepdims=True)
        grad_mu = -2*sigma_inv@grad_mu

        # grad tau
        X_bis = X-mu
        a = np.real(np.einsum('ij,ji->i',
                              np.conjugate(X_bis).T@la.inv(sigma),
                              X_bis))
        a = a.reshape((-1, 1))
        grad_tau = np.real((p*tau-a) * (tau**(-2)))

        # grad sigma
        X_bis = (X-mu) / np.sqrt(tau).T
        grad_sigma = X_bis@X_bis.conj().T
        grad_sigma = -sigma_inv@grad_sigma@sigma_inv

        # grad
        grad = (np.squeeze(grad_mu), grad_tau, grad_sigma)

        return grad

    @pymanopt.function.Callable
    def auto_egrad(mu, tau, sigma):
        res = autograd.grad(cost, argnum=[0, 1, 2])(mu, tau, sigma)
        res = tuple(np.conjugate(res))
        return res

    if autodiff:
        return cost, auto_egrad
    else:
        return cost, egrad


def estimate_location_covariance_texture_RGD(
    X,
    init=None,
    tol=1e-3,
    iter_max=3*int(1e4),
    autodiff=False,
    solver='conjugate'
):
    """ A function that estimates parameters of a compound Gaussian distribution.
        Inputs:
            * X = a matrix of size p*N
            * init = point on manifold to initliase estimation
            * tol = minimum norm of gradient
            * iter_max = maximum number of iterations
            * autodiff = use or not autodiff
            * solver = steepest or conjugate
        Outputs:
            * mu = estimate of location
            * tau = estimate of tau
            * sigma = estimate of covariance """

    # The estimation is done using Riemannian geometry.
    # The manifold is: C^p x SHp++ x (R++)^n
    p, N = X.shape

    # Initialisation
    if init is None:
        mu = np.mean(X, axis=1).reshape((-1, 1))
        sigma = (1/N)*(X-mu)@(X-mu).conj().T
        tau = np.real(la.det(sigma)**(1/p))*np.ones((1, N))
        sigma = sigma/(la.det(sigma)**(1/p))
        init = [mu, tau, sigma]

    init[0] = init[0].reshape(-1)
    init[1] = init[1].reshape((-1, 1))

    cost, egrad = create_cost_egrad_location_covariance_texture(X, autodiff)
    manifold = Product([
        ComplexEuclidean(p),
        StrictlyPositiveVectors(N),
        SpecialHermitianPositiveDefinite(p)])
    problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=0)
    if solver == 'steepest':
        solver = SteepestDescent
    elif solver == 'conjugate':
        solver = ConjugateGradient
    solver = solver(
        maxtime=np.inf,
        maxiter=iter_max,
        mingradnorm=tol,
        minstepsize=0,
        maxcostevals=np.inf,
        logverbosity=2
    )
    Xopt, log = solver.solve(problem, x=init)
    Xopt[0] = Xopt[0].reshape((-1, 1))

    return Xopt[0], Xopt[2], Xopt[1], log


# CLASSES

@make_feature_prototype
def location_covariance_texture_Gaussian(weights=(1, 1, 1), **kwargs):
    p = kwargs['p']
    N = kwargs['N']

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
def location_covariance_texture_Tyler(weights=(1, 1, 1), **kwargs):
    p = kwargs['p']
    N = kwargs['N']

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
def location_covariance_texture_RGD(iter_max=3*int(1e4), weights=(1, 1, 1), **kwargs):
    p = kwargs['p']
    N = kwargs['N']

    name = 'location_covariance_texture_RGD'
    M = (ComplexEuclidean,
         SpecialHermitianPositiveDefinite,
         StrictlyPositiveVectors)
    args_M = {
        'sizes': (p, p, N),
        'weights': weights
    }

    def _estimation(X):
        mu, sigma, tau, _ = estimate_location_covariance_texture_RGD(
            X, iter_max=iter_max)
        return mu, sigma, tau

    return Feature(name, _estimation, M, args_M)
