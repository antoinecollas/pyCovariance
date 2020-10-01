import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

from pyCovariance.features.covariance import distance_covariance_Riemannian
from pyCovariance.features.location_covariance_texture import estimation_location_covariance_texture_RGD, tyler_estimator_location_covariance_normalisedet
from pyCovariance.generation_data import generate_covariance, generate_texture, generate_Toeplitz, sample_compound
from pyCovariance.vectorization import unvech, vech


nb_MC = 10
p = 3
N_max = 10000
nb_points = 5
tol = 1e-10
iter_max = 100

mu = np.random.randn(p, 1) + 1j*np.random.randn(p, 1)
tau_full = generate_texture(N_max)
sigma = generate_covariance(p)
# sigma = generate_Toeplitz(0.1, p)
sigma = (1/np.linalg.det(sigma))**(1/p) * sigma

assert np.abs(np.linalg.det(sigma)-1) < 1e-5

list_n_points = np.geomspace(2*p, N_max, num=nb_points, dtype=np.int)
print(list_n_points)

# Gaussian
mu_errors_g = list()
sigma_errors_g = list()

# Tyler
mu_errors_t = list()
sigma_errors_t = list()

# Riemannian gradient descent
mu_errors_rgd = list()
sigma_errors_rgd = list()

for n in tqdm(list_n_points):
    mu_error_g = 0
    sigma_error_g = 0

    mu_error_t = 0
    sigma_error_t = 0

    mu_error_rgd = 0
    sigma_error_rgd = 0

    tau = tau_full[:n]

    for i in tqdm(range(nb_MC)):
        X = sample_compound(tau, sigma) + mu

        # Gaussian
        mu_g = np.mean(X, axis=1).reshape((-1, 1))
        sigma_g = (1/n)*(X-mu_g)@(X-mu_g).conj().T
        tau_g = np.real(np.linalg.det(sigma_g)**(1/p))*np.ones((n, 1))
        sigma_g = sigma_g/np.real(np.linalg.det(sigma_g)**(1/p))

        mu_error_g += np.linalg.norm(mu-mu_g)**2
        sigma_error_g += distance_covariance_Riemannian(vech(sigma_g), vech(sigma))**2


        # Tyler
        if True:
            mu_0 = np.mean(X, axis=1).reshape((-1, 1))
            tau_0 = np.ones((n, 1))
            sigma_0 = np.eye(p)
            theta_0 = [mu_0, tau_0, sigma_0]
        elif True:
            theta_0 = [mu_g, tau_g, sigma_g]
        
        mu_est, tau_est, sigma_est, _, _ = tyler_estimator_location_covariance_normalisedet(
            X,
            init=theta_0,
            tol=tol,
            iter_max=iter_max
        )
        mu_error_t += np.linalg.norm(mu-mu_est)**2
        sigma_error_t += distance_covariance_Riemannian(vech(sigma_est), vech(sigma))**2

        # gradient descent 
        theta_0 = [mu_g, tau_g, sigma_g]
        
        mu_est, tau_est, sigma_est = estimation_location_covariance_texture_RGD(
            X,
            init=theta_0,
            tol=tol,
            iter_max=iter_max,
            autodiff=False
        )
        mu_error_rgd += np.linalg.norm(mu-mu_est)**2
        sigma_error_rgd += distance_covariance_Riemannian(vech(sigma_est), vech(sigma))**2

    # Gaussian
    mu_errors_g.append(mu_error_g/nb_MC)
    sigma_errors_g.append(sigma_error_g/nb_MC)

    # Tyler
    mu_errors_t.append(mu_error_t/nb_MC)
    sigma_errors_t.append(sigma_error_t/nb_MC)

    # gradient descent 
    mu_errors_rgd.append(mu_error_rgd/nb_MC)
    sigma_errors_rgd.append(sigma_error_rgd/nb_MC)


plt.loglog(list_n_points, mu_errors_g, label='mu - Gaussian')
plt.loglog(list_n_points, mu_errors_t, label='mu - Tyler')
plt.loglog(list_n_points, mu_errors_rgd, label='mu - Riemannian')

plt.loglog(list_n_points, sigma_errors_g, label='sigma - Gaussian')
plt.loglog(list_n_points, sigma_errors_t, label='sigma - Tyler')
plt.loglog(list_n_points, sigma_errors_rgd, label='sigma - Riemannien')

plt.legend()
plt.xlabel('Nombre de points')
plt.ylabel('Erreur d\'estimation')
plt.grid(b=True, which='both')
plt.savefig('tyler.png')

