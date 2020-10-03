import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

from pyCovariance.features.covariance import distance_covariance_Riemannian
from pyCovariance.features.location_covariance_texture import estimation_location_covariance_texture_RGD, tyler_estimator_location_covariance_normalisedet
from pyCovariance.generation_data import generate_covariance, generate_texture, generate_Toeplitz, sample_compound
from pyCovariance.vectorization import unvech, vech


nb_MC = 20
p = 10
N_max = 1000
nb_points = 5
tol = 1e-5
iter_max = 1000

mu = np.random.randn(p, 1) + 1j*np.random.randn(p, 1)
tau_full = generate_texture(N_max)
sigma = generate_covariance(p)
sigma = (1/np.linalg.det(sigma))**(1/p) * sigma

assert np.abs(np.linalg.det(sigma)-1) < 1e-5

list_n_points = np.geomspace(2*p, N_max, num=nb_points, dtype=np.int)
print(list_n_points)

# Gaussian
mu_errors_g = list()
tau_errors_g = list()
sigma_errors_g = list()

# Riemannian gradient descent
mu_errors_rgd = list()
tau_errors_rgd = list()
sigma_errors_rgd = list()

for n in tqdm(list_n_points):
    mu_error_g = list()
    tau_error_g = list()
    sigma_error_g = list()

    mu_error_rgd = list()
    tau_error_rgd = list()
    sigma_error_rgd = list()

    tau = tau_full[:n]

    for i in tqdm(range(nb_MC)):
        X = sample_compound(tau, sigma) + mu

        # Gaussian
        mu_g = np.mean(X, axis=1).reshape((-1, 1))
        sigma_g = (1/n)*(X-mu_g)@(X-mu_g).conj().T
        tau_g = np.real(np.linalg.det(sigma_g)**(1/p))*np.ones((n, 1))
        sigma_g = sigma_g/np.real(np.linalg.det(sigma_g)**(1/p))

        mu_error_g.append(np.linalg.norm(mu-mu_g)**2)
        tau_error_g.append((np.linalg.norm(tau-tau_g)**2)/n)
        sigma_error_g.append(distance_covariance_Riemannian(vech(sigma_g), vech(sigma))**2)

        # Initialisation
        theta_0 = [mu_g, tau_g, sigma_g]

        # gradient descent 
        mu_est, tau_est, sigma_est = estimation_location_covariance_texture_RGD(
            X,
            init=theta_0,
            tol=tol,
            iter_max=iter_max,
            autodiff=False
        )
        mu_error_rgd.append(np.linalg.norm(mu-mu_est)**2)
        tau_error_rgd.append((np.linalg.norm(tau-tau_est)**2)/n)
        sigma_error_rgd.append(distance_covariance_Riemannian(vech(sigma_est), vech(sigma))**2)

    # Gaussian
    mu_errors_g.append(np.mean(mu_error_g))
    tau_errors_g.append(np.mean(tau_error_g))
    sigma_errors_g.append(np.mean(sigma_error_g))

    # gradient descent 
    mu_errors_rgd.append(np.mean(mu_error_rgd))
    tau_errors_rgd.append(np.mean(tau_error_rgd))
    sigma_errors_rgd.append(np.mean(sigma_error_rgd))


plt.loglog(list_n_points, mu_errors_g, color='b', marker='^', label='mu - Gaussian')
plt.loglog(list_n_points, mu_errors_rgd, color='g', marker='^', label='mu - Riemannian')
plt.legend()
plt.xlabel('Number of data points')
plt.ylabel('MSE on mu')
plt.grid(b=True, which='both')
plt.savefig('results/location_tyler_mu.png')
plt.clf()

plt.loglog(list_n_points, tau_errors_g, color='b', marker='+', label='tau - Gaussian')
plt.loglog(list_n_points, tau_errors_rgd, color='g', marker='+', label='tau - Riemannian')
plt.legend()
plt.xlabel('Number of data points')
plt.ylabel('MSE on tau')
plt.grid(b=True, which='both')
plt.savefig('results/location_tyler_tau.png')
plt.clf()

plt.loglog(list_n_points, sigma_errors_g, color='b', marker='*', label='sigma - Gaussian')
plt.loglog(list_n_points, sigma_errors_rgd, color='g', marker='*', label='sigma - Riemannian')
plt.legend()
plt.xlabel('Number of data points')
plt.ylabel('MSE on Sigma')
plt.grid(b=True, which='both')
plt.savefig('results/location_tyler_sigma.png')
plt.clf()
