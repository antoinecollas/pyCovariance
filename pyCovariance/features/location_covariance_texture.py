import autograd.numpy as np
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import ComplexEuclidean, Product, StrictlyPositiveVectors, SpecialHermitianPositiveDefinite
from pymanopt.solvers import SteepestDescent
import warnings

from .base import BaseClassFeatures
from ..vectorization import *

########## ESTIMATION ##########

def tyler_estimator_location_covariance_normalisedet(X, init=None, tol=0.001, iter_max=20):
    """ A function that computes the Tyler Fixed Point Estimator for location and covariance estimation.
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * init = point on manifold to initliase estimation
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
        tau = np.linalg.det(sigma)**(1/p)*np.ones((1, N))
        sigma = sigma/(np.linalg.det(sigma)**(1/p))
    else:
        mu, tau, sigma = init

    mu = mu.reshape((-1, 1))
    tau = tau.reshape((1, -1))

    delta = np.inf # Distance between two iterations
    iteration = 0

    while (delta>tol) and (iteration<iter_max):
        # compute tau
        tau_new = np.real(np.einsum('ij,ji->i', np.conjugate(X-mu).T@np.linalg.inv(sigma), X-mu))
        
        # compute mu (location)
        mu_new = (1/np.sum(1/tau)) * np.sum(X/tau, axis=1).reshape((-1, 1))

        # compute sigma
        X_bis = (X-mu) / np.sqrt(tau)
        sigma_new = (1/N) * X_bis@X_bis.conj().T

        # imposing det constraint: det(sigma_new) = 1
        sigma_new = sigma_new/(np.linalg.det(sigma_new)**(1/p))
 
        # condition for stopping
        delta = np.linalg.norm(sigma_new - sigma, 'fro') / np.linalg.norm(sigma, 'fro')
        iteration = iteration + 1

        # updating
        tau = tau_new
        mu = mu_new
        sigma = sigma_new
    
    if iteration == iter_max:
        warnings.warn('Estimation algorithm did not converge')

    return (mu, tau, sigma, delta, iteration)


def create_cost_egrad_location_covariance_texture(X, autodiff=False):
    @pymanopt.function.Callable
    def cost(mu, tau, sigma):
        p, N = X.shape
        
        # compute quadratic form
        mu = mu.reshape((-1, 1))
        X_bis = X-mu
        Q = np.real(np.einsum('ij,ji->i', np.conjugate(X_bis).T@np.linalg.inv(sigma), X_bis))
    
        tau = np.squeeze(tau)
        L = p*np.log(tau) + Q/tau
        L = np.real(np.sum(L))

        return L

    @pymanopt.function.Callable
    def egrad(mu, tau, sigma):
        p, N = X.shape
        sigma_inv = np.linalg.inv(sigma)

        # grad mu
        mu = mu.reshape((-1, 1))
        grad_mu = (X-mu)/tau.T
        grad_mu = np.sum(grad_mu, axis=1, keepdims=True)
        grad_mu = -2*sigma_inv@grad_mu

        # grad tau
        X_bis = X-mu
        a = np.real(np.einsum('ij,ji->i', np.conjugate(X_bis).T@np.linalg.inv(sigma), X_bis)).reshape((-1, 1))
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
        res = tuple(np.conjugate(autograd.grad(cost, argnum=[0, 1, 2])(mu, tau, sigma)))
        return res

    if autodiff:
        return cost, auto_egrad
    else:
        return cost, egrad


def estimation_location_covariance_texture_RGD(X, init=None, autodiff=False):
    """ A function that estimates parameters of a compound Gaussian distribution.
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * init = point on manifold to initliase estimation
            * autodiff = use or not autodiff
        Outputs:
            * mu = estimate of location
            * sigma = estimate of covariance
            * tau = estimate of covariance """
    
    # The estimation is done using Riemannian geometry. The manifold is: C^p x (R++)^n x SHp++
    p, N = X.shape
    init[0] = init[0].reshape(-1) 
    init[1] = init[1].reshape((N, 1)) 

    cost, egrad = create_cost_egrad_location_covariance_texture(X, autodiff)
    manifold = Product([ComplexEuclidean(p), StrictlyPositiveVectors(N), SpecialHermitianPositiveDefinite(p)])
    problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=0)
    solver = SteepestDescent()
    Xopt = solver.solve(problem, x=init)
    Xopt[0] = Xopt[0].reshape((-1, 1))
    return Xopt

##########  DISTANCE  ##########

##########   MEAN     ##########

##########  CLASSES   ##########

