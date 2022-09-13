import autograd
import autograd.numpy as np
import autograd.numpy.linalg as la
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import\
        ComplexEuclidean,\
        HermitianPositiveDefinite,\
        SpecialHermitianPositiveDefinite
from pymanopt.solvers import ConjugateGradient, SteepestDescent
import warnings

from .base import Feature, make_feature_prototype, Product
from ..manifolds import\
        ComplexCompoundGaussianIGConstrainedScatter,\
        ComplexCompoundGaussianIGConstrainedTexture,\
        ComplexCompoundGaussianMLConstrainedTexture,\
        SpecialStrictlyPositiveVectors,\
        StrictlyPositiveVectors


# Different estimators are proposed.
# They are based on the compound Gaussian distribution,
# i.e, x_i \sim N(\mu, \tau_i \Sigma).
# Since there is an indeterminacy between the textures \tau_i and
# the scatter matrix \Sigma, a constraint must be added.
# Two constraints are used :
# - constraint of unitary determinant of the scatter matrix: det(\Sigma) = 1,
# - constraint of unitary product of the textures: \prod \tau_i = 1.

# Gaussian estimation


def Gaussian_estimation_constrained_scatter(X):
    """ Function that computes Gaussian estimators: sample mean, SCM,
        and constant textures with a unitary determinant constraint
        on the scatter matrix.
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


def Gaussian_estimation_constrained_texture(X):
    """ Function that computes Gaussian estimators: sample mean, SCM,
        and constant textures with a unitary product constraint
        on the textures.
        Inputs:
            * X = a matrix of size p*N
        Outputs:
            * mu = estimate of location
            * sigma = estimate of covariance"""
    p, N = X.shape
    mu = np.mean(X, axis=1).reshape((-1, 1))
    sigma = (1/N) * (X-mu) @ (X-mu).conj().T
    tau = np.ones((N, 1))
    return mu, sigma, tau


# ESTIMATION

def _Tyler_estimator_unknown_helper(X, init, tol, iter_max):

    # Initialisation
    p, N = X.shape
    if init is None:
        mu, sigma, _ = Gaussian_estimation_constrained_texture(X)
    else:
        mu, sigma, _ = init

    mu = mu.reshape((-1, 1))

    delta = np.inf  # Distance between two iterations
    iteration = 0
    while (delta > tol) and (iteration < iter_max):
        # compute mu (location)
        tau = np.einsum('ij,ji->i', np.conjugate(X-mu).T@la.inv(sigma), X-mu)
        tau = np.real(tau) / p
        t = np.sqrt(tau)
        mu = (1/np.sum(1/t)) * np.sum(X/t, axis=1).reshape((-1, 1))

        # compute sigma
        tau = np.einsum('ij,ji->i', np.conjugate(X-mu).T@la.inv(sigma), X-mu)
        tau = np.real(tau) / p
        t = np.sqrt(tau)
        X_bis = (X-mu) / t
        sigma_new = (1/N) * X_bis@X_bis.conj().T

        # condition for stopping
        delta = la.norm(sigma_new - sigma, 'fro') / la.norm(sigma, 'fro')
        iteration = iteration + 1

        # updating
        sigma = sigma_new

    # recomputing tau
    tau = np.einsum('ij,ji->i', np.conjugate(X-mu).T@la.inv(sigma), X-mu)
    tau = np.real(tau) / p

    if iteration == iter_max:
        warnings.warn('Estimation algorithm did not converge')

    mu = mu.reshape((-1, 1))
    tau = tau.reshape((-1, 1))

    return mu, sigma, tau, delta, iteration


def Tyler_estimator_unknown_location_constrained_scatter(
    X,
    init=None,
    tol=0.001,
    iter_max=100
):
    """ A function that computes the Tyler Fixed Point Estimator for
        location and covariance estimation with a constraint of unitary
        determinant on the scatter matrix.
        Inputs:
            * X = a matrix of size p*N
            * init = point on manifold to initialise estimation
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * mu = estimate of location
            * sigma = estimate of covariance
            * delta = the final distance between two iterations
            * iteration = number of iterations until convergence """

    mu, sigma, tau, delta, iteration = _Tyler_estimator_unknown_helper(
        X, init, tol, iter_max)

    # imposing det constraint: det(sigma_new) = 1
    p, _ = X.shape
    c = np.real(la.det(sigma)**(1/p))
    sigma = sigma/c
    tau = c*tau

    return mu, sigma, tau, delta, iteration


def create_cost_egrad_ehess_nll_CG_constrained_scatter(
    X,
    autodiff=False
):
    """ A function that returns functions to compute quantities associated to
        the compound Gaussian distribution with a constraint of unit
        determinant of the scatter matrix.
        - the negative log-likelihood of the compound Gaussian distribution,
        - the Euclidean gradient of the negative log-likelihood,
        - the Euclidean Hessian of the negative log-likelihood.
        Inputs:
            * X = a matrix of size p*N
            * autodiff = boolean to use autodiff or closed form grad/hess
        Outputs:
            * cost = negative log-likelihood
            * egrad = Euclidean gradient of the negative log-likelihood
            * ehess = Euclidean Hessian of the negative log-likelihood"""

    @pymanopt.function.Callable
    def cost(mu, sigma, tau):
        p, N = X.shape

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
        res = tuple([np.conjugate(r) for r in res])
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
        ehess = tuple([np.conjugate(r) for r in ehess])
        return ehess

    if autodiff:
        return cost, auto_egrad, auto_ehess
    else:
        return cost, egrad, ehess


def create_cost_egrad_ehess_nll_CG_constrained_texture(
    X,
    reg_type='AF',
    reg_beta=0,
    reg_kappa='trace_SCM',
    autodiff=False
):
    """ A function that returns functions to compute quantities associated to
        the compound Gaussian distribution with a constraint of
        unitary product on the textures.
        - the negative log-likelihood of the compound Gaussian distribution,
        - the Euclidean gradient of the negative log-likelihood,
        - the Euclidean Hessian of the negative log-likelihood.
        Inputs:
            * X = a matrix of size p*N
            * reg_type = type of regularization, e.g. L1, L2, etc...
            * reg_beta = intensity of regularization, the higher the beta,
            the greater the regularization
            * reg_kappa = the estimator is regularized towards kappa*identity
            * autodiff = boolean to use autodiff or closed form grad/hess
        Outputs:
            * cost = negative log-likelihood
            * egrad = Euclidean gradient of the negative log-likelihood
            * ehess = Euclidean Hessian of the negative log-likelihood"""
    beta = reg_beta
    kappa = reg_kappa
    if kappa == 'trace_SCM':
        p, _ = X.shape
        _, scm, _ = Gaussian_estimation_constrained_texture(X)
        kappa = np.real(np.trace(scm))/p
    elif (type(kappa) in [float, int]) and (kappa > 0):
        pass
    else:
        s = 'Wrong value of "kappa" regularization hyperparameter.'
        raise ValueError(s)

    def reg_L1(x):
        tmp = np.abs(x**(-1) - kappa**(-1))
        return np.sum(tmp)

    def reg_L2(x):
        tmp = (x**(-1) - kappa**(-1))**2
        return np.sum(tmp)

    def reg_AF(x):
        tmp = (np.log(x) - np.log(kappa))**2
        return np.sum(tmp)

    def reg_BW(x):
        tmp = (x**(-0.5) - kappa**(-0.5))**2
        return np.sum(tmp)

    def reg_KL(x):
        tmp = kappa*(x**(-1)) + np.log(x)
        return np.sum(tmp)

    if reg_type == 'L1':
        reg_fct = reg_L1
    elif reg_type == 'L2':
        reg_fct = reg_L2
    elif reg_type == 'AF':
        reg_fct = reg_AF
    elif reg_type == 'BW':
        reg_fct = reg_BW
    elif reg_type == 'KL':
        reg_fct = reg_KL
    else:
        s = 'Regularizations available: L1, L2, AF, BW, KL.'
        raise ValueError(s)

    @pymanopt.function.Callable
    def cost(mu, sigma, tau):
        p, N = X.shape

        X_bis = X-mu
        Q = np.real(np.einsum('ij,ji->i',
                              np.conjugate(X_bis).T@la.inv(sigma),
                              X_bis))
        tau = np.squeeze(tau)
        L = np.real(np.sum(Q/tau))
        L = N*np.log(np.real(la.det(sigma))) + L

        # regularization
        eigvals, _ = la.eigh(sigma)
        tmp = np.einsum('i,j->ij', tau, np.real(eigvals)).reshape(-1)
        L = L + beta*reg_fct(tmp)
        L = L/(p*N)

        return L

    @pymanopt.function.Callable
    def egrad(mu, sigma, tau):
        raise NotImplementedError

    @pymanopt.function.Callable
    def ehess(mu, sigma, tau, xi_mu, xi_sigma, xi_tau):
        raise NotImplementedError

    @pymanopt.function.Callable
    def auto_egrad(mu, sigma, tau):
        res = autograd.grad(cost, argnum=[0, 1, 2])(mu, sigma, tau)
        res = tuple([np.conjugate(r) for r in res])
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
        ehess = tuple([np.conjugate(r) for r in ehess])
        return ehess

    if autodiff:
        return cost, auto_egrad, auto_ehess
    else:
        return cost, egrad, ehess


def estimate_compound_Gaussian_constrained_scatter(
    X,
    init=None,
    information_geometry=True,
    reg=0,
    min_grad_norm=1e-5,
    min_step_size=1e-8,
    iter_max=500,
    time_max=np.inf,
    autodiff=False,
    solver='steepest'
):
    """ A function that estimates parameters of a
        compound Gaussian distribution with a constraint of
        unitay determinant on the scatter matrix.
        Inputs:
            * X = a matrix of size p*N
            * init = point on manifold to initliase estimation
            * information_geometry = use manifold of Compound distribution
            * reg = regularization of cost function
            * min_grad_norm = minimum norm of gradient
            * min_step_size = minimum step size
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

    if reg != 0:
        raise NotImplementedError

    # Initialisation
    if init is None:
        init = list(Gaussian_estimation_constrained_scatter(X))

    init[0] = init[0].reshape((-1, 1))
    init[2] = init[2].reshape((-1, 1))

    # cost, egrad, ehess
    cost, egrad, ehess = create_cost_egrad_ehess_nll_CG_constrained_scatter(
        X,
        autodiff
    )

    # solver
    if solver == 'steepest':
        solver = SteepestDescent
    elif solver == 'conjugate':
        solver = ConjugateGradient
    else:
        s = 'Solvers available: steepest, conjugate.'
        raise ValueError(s)
    solver = solver(maxtime=time_max, maxiter=iter_max,
                    mingradnorm=min_grad_norm, minstepsize=min_step_size,
                    maxcostevals=np.inf, logverbosity=2)

    # manifold
    if information_geometry:
        manifold = ComplexCompoundGaussianIGConstrainedScatter(p, N)
    else:
        manifold = Product([ComplexEuclidean(p, 1),
                            SpecialHermitianPositiveDefinite(p),
                            StrictlyPositiveVectors(N)])

    problem = Problem(manifold=manifold, cost=cost, egrad=egrad,
                      ehess=ehess, verbosity=0)
    Xopt, log = solver.solve(problem, x=init)

    return Xopt[0], Xopt[1], Xopt[2], log


def estimate_compound_Gaussian_constrained_texture(
    X,
    init=None,
    information_geometry=True,
    reg_type='AF',
    reg_beta=0,
    reg_kappa='trace_SCM',
    min_grad_norm=1e-5,
    min_step_size=1e-8,
    iter_max=500,
    time_max=np.inf,
    autodiff=True,
    solver='steepest'
):
    """ A function that estimates parameters of a
        compound Gaussian distribution with a constraint of
        unitary product on the textures.
        Inputs:
            * X = a matrix of size p*N
            * init = point on manifold to initliase estimation
            * information_geometry = use manifold of Compound distribution
            * reg_type = type of regularization, e.g. L1, L2, etc...
            * reg_beta = intensity of regularization, the higher the beta,
            the greater the regularization
            * reg_kappa = the estimator is regularized towards kappa*identity
            * min_grad_norm = minimum norm of gradient
            * min_step_size = minimum step size
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
        kappa = reg_kappa
        if kappa == 'trace_SCM':
            p, _ = X.shape
            _, scm, _ = Gaussian_estimation_constrained_texture(X)
            kappa = np.real(np.trace(scm))/p
        elif (type(kappa) in [float, int]) and (kappa > 0):
            pass
        else:
            s = 'Wrong value of "kappa" regularization hyperparameter.'
            raise ValueError(s)
        init = list()
        init.append(np.mean(X, axis=1, keepdims=True))
        init.append(kappa*np.eye(p, dtype=X.dtype))
        init.append(np.ones((N, 1)))

    init[0] = init[0].reshape((-1, 1))
    init[2] = init[2].reshape((-1, 1))

    # cost, egrad, ehess
    cost, egrad, ehess = create_cost_egrad_ehess_nll_CG_constrained_texture(
        X,
        reg_type=reg_type,
        reg_beta=reg_beta,
        reg_kappa=reg_kappa,
        autodiff=autodiff
    )

    # solver
    if solver == 'steepest':
        solver = SteepestDescent
    elif solver == 'conjugate':
        solver = ConjugateGradient
    else:
        s = 'Solvers available: steepest, conjugate.'
        raise ValueError(s)
    solver = solver(maxtime=time_max, maxiter=iter_max,
                    mingradnorm=min_grad_norm, minstepsize=min_step_size,
                    maxcostevals=np.inf, logverbosity=2)

    # manifold
    if information_geometry:
        manifold = ComplexCompoundGaussianIGConstrainedTexture(p, N)
    else:
        manifold = Product([ComplexEuclidean(p, 1),
                            HermitianPositiveDefinite(p),
                            SpecialStrictlyPositiveVectors(N)])

    problem = Problem(manifold=manifold, cost=cost, egrad=egrad,
                      ehess=ehess, verbosity=0)
    Xopt, log = solver.solve(problem, x=init)

    # mu, sigma, tau = Xopt[0], Xopt[1], Xopt[2]
    # mean = np.mean(X, axis=1, keepdims=True)
    # print('la.norm(mean-mu)=', la.norm(mean-mu))
    # print('kappa=', kappa)
    # eigvals, _ = la.eigh(sigma)
    # print('eigvals=', eigvals)
    # print('np.min(tau)=', np.min(tau))
    # print('np.max(tau)=', np.max(tau))

    return Xopt[0], Xopt[1], Xopt[2], log


# CLASSES WITH CONSTRAINT ON SCATTER MATRIX

@make_feature_prototype
def location_covariance_texture_Gaussian_constrained_scatter(
    weights=(1, 1, 1),
    min_grad_norm_mean=5*1e-6,
    p=None,
    N=None
):
    name = 'Location_covariance_texture_Gaussian_constrained_scatter'
    M = (ComplexEuclidean,
         SpecialHermitianPositiveDefinite,
         StrictlyPositiveVectors)
    args_M = {
        'sizes': (p, p, N),
        'weights': weights
    }

    return Feature(name, Gaussian_estimation_constrained_scatter, M, args_M,
                   min_grad_norm_mean=min_grad_norm_mean)


@make_feature_prototype
def location_covariance_texture_Tyler_constrained_scatter(
    weights=(1, 1, 1),
    min_grad_norm_mean=5*1e-6,
    p=None,
    N=None
):
    name = 'Location_covariance_texture_Tyler_constrained_scatter'
    M = (ComplexEuclidean,
         SpecialHermitianPositiveDefinite,
         StrictlyPositiveVectors)
    args_M = {
        'sizes': (p, p, N),
        'weights': weights
    }

    def _estimation(X):
        mu, sigma, tau, _, _ =\
                Tyler_estimator_unknown_location_constrained_scatter(X)
        return mu, sigma, tau

    return Feature(name, _estimation, M, args_M,
                   min_grad_norm_mean=min_grad_norm_mean)


@make_feature_prototype
def location_covariance_texture_constrained_scatter(
    iter_max=3*int(1e4),
    information_geometry=True,
    solver='steepest',
    weights=(1, 1, 1),
    min_grad_norm_mean=5*1e-6,
    p=None,
    N=None
):
    name = 'Location_covariance_texture_constrained_scatter'
    M = (ComplexEuclidean,
         SpecialHermitianPositiveDefinite,
         StrictlyPositiveVectors)
    args_M = {
        'sizes': (p, p, N),
        'weights': weights
    }

    def _estimation(X):
        mu, sigma, tau, _ = estimate_compound_Gaussian_constrained_scatter(
            X,
            information_geometry=information_geometry,
            iter_max=iter_max,
            solver=solver
        )
        return mu, sigma, tau

    return Feature(name, _estimation, M, args_M,
                   min_grad_norm_mean=min_grad_norm_mean)


# CLASSES WITH CONSTRAINT ON TEXTURE


@make_feature_prototype
def location_covariance_texture_Gaussian_constrained_texture(
    weights=(1, 1, 1),
    min_grad_norm_mean=5*1e-6,
    p=None,
    N=None
):
    name = 'Location_covariance_texture_Gaussian_constrained_texture'
    M = (ComplexEuclidean,
         HermitianPositiveDefinite,
         SpecialStrictlyPositiveVectors)
    args_M = {
        'sizes': (p, p, N),
        'weights': weights
    }

    return Feature(name, Gaussian_estimation_constrained_texture, M, args_M,
                   min_grad_norm_mean=min_grad_norm_mean)


@make_feature_prototype
def location_covariance_texture_constrained_texture(
    iter_max=300,
    information_geometry=True,
    solver='steepest',
    reg_type='AF',
    reg_beta=0,
    reg_kappa='trace_SCM',
    weights=(1, 1, 1),
    min_grad_norm_mean=5*1e-6,
    p=None,
    N=None
):
    name = 'Location_covariance_texture_constrained_texture'
    M = (ComplexEuclidean,
         HermitianPositiveDefinite,
         SpecialStrictlyPositiveVectors)
    args_M = {
        'sizes': (p, p, N),
        'weights': weights
    }

    def _estimation(X):
        mu, sigma, tau, _ = estimate_compound_Gaussian_constrained_texture(
            X,
            information_geometry=information_geometry,
            iter_max=iter_max,
            solver=solver,
            reg_type=reg_type,
            reg_beta=reg_beta,
            reg_kappa=reg_kappa
        )
        return mu, sigma, tau

    return Feature(name, _estimation, M, args_M,
                   min_grad_norm_mean=min_grad_norm_mean)


@make_feature_prototype
def location_covariance_texture_constrained_texture_triangle(
    iter_max=300,
    information_geometry=True,
    solver='steepest',
    reg_type='AF',
    reg_beta=0,
    reg_kappa='trace_SCM',
    weights=(1, 1),
    min_grad_norm_mean=5*1e-6,
    p=None,
    N=None
):
    name = 'Location_covariance_texture_constrained_texture_triangle'
    M = ComplexCompoundGaussianMLConstrainedTexture
    args_M = {
        'sizes': (p, N),
        'weights': weights
    }

    def _estimation(X):
        mu, sigma, tau, _ = estimate_compound_Gaussian_constrained_texture(
            X,
            information_geometry=information_geometry,
            iter_max=iter_max,
            solver=solver,
            reg_type=reg_type,
            reg_beta=reg_beta,
            reg_kappa=reg_kappa
        )
        return mu, sigma, tau

    # init mean
    # argcosh(1 + x) is not differentiable at x = 0.
    # Hence we have to provide an initialization that is not a (mu, sigma)
    # among the parameters to cluster/classify.
    def init_mean(X):
        theta = [
            np.mean(X[0], axis=0),
            np.mean(X[1], axis=0),
            np.ones_like(X[2][0, :, :])
        ]
        return theta

    return Feature(name, _estimation, M, args_M, init_mean,
                   min_grad_norm_mean=min_grad_norm_mean)


@make_feature_prototype
def location_covariance_texture_constrained_texture_div_alpha(
    iter_max=300,
    information_geometry=True,
    solver='steepest',
    reg_type='AF',
    reg_beta=0,
    reg_kappa='trace_SCM',
    alpha=0.5,
    div_alpha_real_case=False,
    symmetrize_div=False,
    min_grad_norm_mean=5*1e-6,
    p=None,
    N=None
):
    name = 'Location_covariance_texture_constrained_texture_'
    name += 'div_alpha_'
    if symmetrize_div:
        name += 'sym_'
    name += str(alpha)
    name += '_reg_' + reg_type + '_' + str(reg_beta)

    if symmetrize_div:
        if div_alpha_real_case:
            class M(ComplexCompoundGaussianIGConstrainedTexture):
                def dist(self, x1, x2):
                    return self.div_alpha_sym_real_case(x1, x2)
        else:
            class M(ComplexCompoundGaussianIGConstrainedTexture):
                def dist(self, x1, x2):
                    return self.div_alpha_sym(x1, x2)
    else:
        if div_alpha_real_case:
            class M(ComplexCompoundGaussianIGConstrainedTexture):
                def dist(self, x1, x2):
                    return self.div_alpha_real_case(x1, x2)
        else:
            class M(ComplexCompoundGaussianIGConstrainedTexture):
                def dist(self, x1, x2):
                    return self.div_alpha(x1, x2)

    args_M = {
        'sizes': (p, N),
        'alpha': alpha
    }

    def _estimation(X):
        mu, sigma, tau, _ = estimate_compound_Gaussian_constrained_texture(
            X,
            information_geometry=information_geometry,
            iter_max=iter_max,
            solver=solver,
            reg_type=reg_type,
            reg_beta=reg_beta,
            reg_kappa=reg_kappa
        )
        return mu, sigma, tau

    # init mean
    # Numerically it is more stable to initialize the mean this way
    # rather than choosing a point randomly among the points of the cluster.
    def init_mean(X):
        # geometric mean of the textures
        tmp = np.exp(np.mean(np.log(X[2]), axis=0))
        theta = [
            np.mean(X[0], axis=0),
            np.mean(X[1], axis=0),
            tmp
        ]
        return theta

    return Feature(name, _estimation, M, args_M, init_mean,
                   min_grad_norm_mean=min_grad_norm_mean)
