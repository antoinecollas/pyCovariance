import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

from pyCovariance.features.covariance import distance_covariance_Riemannian
from pyCovariance.features.covariance_texture import distance_texture_Riemannian
from pyCovariance.features.location_covariance_texture import estimation_location_covariance_texture_RGD, estimation_location_covariance_texture_MLE, tyler_estimator_location_covariance_normalisedet
from pyCovariance.generation_data import generate_covariance, generate_texture, generate_Toeplitz, sample_compound
from pyCovariance.vectorization import unvech, vech


nb_MC = 20
p = 10
nb_points = 50
tol = 1e-5
iter_max = 10000

mu = np.random.randn(p, 1) + 1j*np.random.randn(p, 1)
tau = generate_texture(nb_points)
sigma = generate_covariance(p)
sigma = (1/np.linalg.det(sigma))**(1/p) * sigma

assert np.abs(np.linalg.det(sigma)-1) < 1e-5

# MLE
mu_errors_m = list()
tau_errors_m = list()
sigma_errors_m = list()

# Riemannian gradient descent
mu_errors_rgd = list()
tau_errors_rgd = list()
sigma_errors_rgd = list()

for i in tqdm(range(nb_MC)):
    # MLE
    mu_error_m = list()
    tau_error_m = list()
    sigma_error_m = list()

    # Riemannian gradient descent
    mu_error_rgd = list()
    tau_error_rgd = list()
    sigma_error_rgd = list()

    X = sample_compound(tau, sigma) + mu

    # Gaussian
    mu_g = np.mean(X, axis=1).reshape((-1, 1))
    sigma_g = (1/nb_points)*(X-mu_g)@(X-mu_g).conj().T
    tau_g = np.real(np.linalg.det(sigma_g)**(1/p))*np.ones((nb_points, 1))
    sigma_g = sigma_g/np.real(np.linalg.det(sigma_g)**(1/p))

    # Initialisation
    theta_0 = [mu_g, tau_g, sigma_g]

    # MLE
    mu_est, tau_est, sigma_est, _, _ = estimation_location_covariance_texture_MLE(
        X,
        init=theta_0,
        tol=tol,
        iter_max=iter_max,
    )
    mu_error_m.append(np.linalg.norm(mu-mu_est)**2)
    tau_error_m.append((distance_texture_Riemannian(tau, tau_est)**2)/nb_points)
    sigma_error_m.append(distance_covariance_Riemannian(vech(sigma_est), vech(sigma))**2)
    
    if False:
        # gradient descent 
        mu_est, tau_est, sigma_est, _ = estimation_location_covariance_texture_RGD(
            X,
            init=theta_0,
            tol=tol,
            iter_max=iter_max,
            autodiff=False
        )
        mu_error_rgd.append(np.linalg.norm(mu-mu_est)**2)
        tau_error_rgd.append((distance_texture_Riemannian(tau, tau_est)**2)/n)
        sigma_error_rgd.append(distance_covariance_Riemannian(vech(sigma_est), vech(sigma))**2)

    # MLE
    mu_errors_m.append(np.mean(mu_error_m))
    tau_errors_m.append(np.mean(tau_error_m))
    sigma_errors_m.append(np.mean(sigma_error_m))

    if False:
        # gradient descent 
        mu_errors_rgd.append(np.mean(mu_error_rgd))
        tau_errors_rgd.append(np.mean(tau_error_rgd))
        sigma_errors_rgd.append(np.mean(sigma_error_rgd))

path = 'results/location_cov_text_MLE'

plt.loglog(list_n_points, mu_errors_m, marker='+', color='g', label='mu - MLE')
plt.legend()
plt.xlabel('Number of data points')
plt.ylabel('MSE on mu')
plt.grid(b=True, which='both')
plt.savefig(path+'_mu.png')
plt.clf()

plt.loglog(list_n_points, tau_errors_m, marker='.', color='r', label='tau - MLE')
plt.legend()
plt.xlabel('Number of data points')
plt.ylabel('MSE on tau')
plt.grid(b=True, which='both')
plt.savefig(path+'_tau.png')
plt.clf()

plt.loglog(list_n_points, sigma_errors_m, marker='s', color='c', label='sigma - MLE')
plt.legend()
plt.xlabel('Number of data points')
plt.ylabel('MSE on sigma')
plt.grid(b=True, which='both')
plt.savefig(path+'_sigma.png')
plt.clf()
