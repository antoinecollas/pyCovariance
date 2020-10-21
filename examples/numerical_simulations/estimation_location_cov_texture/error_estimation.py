import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import tikzplotlib

from pyCovariance.features.covariance import distance_covariance_Riemannian
from pyCovariance.features.covariance_texture import distance_texture_Riemannian, tyler_estimator_covariance_normalisedet
from pyCovariance.features.location_covariance_texture import estimation_location_covariance_texture_RGD, tyler_estimator_location_covariance_normalisedet
from pyCovariance.generation_data import generate_covariance, generate_texture, generate_Toeplitz, sample_compound
from pyCovariance.vectorization import unvech, vech


path = 'results/location_tyler_vs_grad'
path_data = os.path.join(path+'_data')
if not os.path.exists(path):
    os.makedirs(path_data, exist_ok=True)

np.random.seed(123)

nb_MC = 2000
p = 10
N_min = 2*p
N_max = 50*p
nb_points = 10
tol = 1e-8
iter_max = 10000
alpha = 10

mu = alpha + (1/np.sqrt(2))*(np.random.randn(p, 1) + 1j*np.random.randn(p, 1))
tau_full = generate_texture(N_max)
sigma = generate_covariance(p)
sigma = (1/np.linalg.det(sigma))**(1/p) * sigma

# save data
np.save(os.path.join(path_data, 'mu.npy'), mu)
np.save(os.path.join(path_data, 'tau.npy'), tau_full)
np.save(os.path.join(path_data, 'sigma.npy'), sigma)

assert np.abs(np.linalg.det(sigma)-1) < 1e-5

list_n_points = np.geomspace(N_min, N_max, num=nb_points, dtype=np.int)
print('N=', list_n_points)

# Gaussian
mu_errors_g = list()
tau_errors_g = list()
sigma_errors_g = list()

# Tyler location known
mu_errors_t = list()
tau_errors_t = list()
sigma_errors_t = list()

# Tyler
mu_errors_t2 = list()
tau_errors_t2 = list()
sigma_errors_t2 = list()

# Riemannian gradient descent
mu_errors_rgd = list()
tau_errors_rgd = list()
sigma_errors_rgd = list()

for n in tqdm(list_n_points):
    # Gaussian
    mu_error_g = list()
    tau_error_g = list()
    sigma_error_g = list()

    # Tyler location known
    mu_error_t = list()
    tau_error_t = list()
    sigma_error_t = list()

    # Tyler
    mu_error_t2 = list()
    tau_error_t2 = list()
    sigma_error_t2 = list()

    # Riemannian gradient descent
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

        # Tyler location known
        X_centered = X-mu
        X_centered_filtered = X_centered[:, np.linalg.norm(X_centered, axis=0)>1e-8]
        _, sigma_est, _, _ = tyler_estimator_covariance_normalisedet(
            X_centered_filtered,
            init=[theta_0[1], theta_0[2]],
            tol=tol,
            iter_max=iter_max,
        )
        mu_est = mu
        tau_est = (1/p) * np.real(np.einsum('ij,ji->i', np.conjugate(X_centered).T@np.linalg.inv(sigma_est), X_centered))
        mu_error_t.append(np.linalg.norm(mu-mu_est)**2)
        tau_error_t.append((distance_texture_Riemannian(tau, tau_est)**2)/n)
        sigma_error_t.append(distance_covariance_Riemannian(vech(sigma_est), vech(sigma))**2)

        # Tyler
        mu_est, tau_est, sigma_est, _, _ = tyler_estimator_location_covariance_normalisedet(
            X,
            init=theta_0,
            tol=tol,
            iter_max=iter_max,
        )
        mu_error_t2.append(np.linalg.norm(mu-mu_est)**2)
        tau_error_t2.append((distance_texture_Riemannian(tau, tau_est)**2)/n)
        sigma_error_t2.append(distance_covariance_Riemannian(vech(sigma_est), vech(sigma))**2)

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


    # Gaussian
    mu_errors_g.append(np.mean(mu_error_g))
    tau_errors_g.append(np.mean(tau_error_g))
    sigma_errors_g.append(np.mean(sigma_error_g))

    # Tyler location known
    mu_errors_t.append(np.mean(mu_error_t))
    tau_errors_t.append(np.mean(tau_error_t))
    sigma_errors_t.append(np.mean(sigma_error_t))

    # Tyler
    mu_errors_t2.append(np.mean(mu_error_t2))
    tau_errors_t2.append(np.mean(tau_error_t2))
    sigma_errors_t2.append(np.mean(sigma_error_t2))

    # gradient descent 
    mu_errors_rgd.append(np.mean(mu_error_rgd))
    tau_errors_rgd.append(np.mean(tau_error_rgd))
    sigma_errors_rgd.append(np.mean(sigma_error_rgd))


plt.loglog(list_n_points, mu_errors_g, marker='*', color='b', label='mu - Gaussian')
plt.loglog(list_n_points, mu_errors_t2, marker='+', color='g', label='mu - Tyler')
plt.loglog(list_n_points, mu_errors_rgd, marker='.', color='r', label='mu - Riemannian')
plt.legend()
plt.xlabel('Number of data points')
plt.ylabel('MSE on mu')
plt.grid(b=True, which='both')
plt.savefig(path+'_mu.png')
tikzplotlib.save(path+'_mu.tex')
plt.clf()

plt.loglog(list_n_points, tau_errors_g, marker='*', color='b', label='tau - Gaussian')
plt.loglog(list_n_points, tau_errors_t, marker='^', color='k', label='tau - Tyler - location known')
plt.loglog(list_n_points, tau_errors_t2, marker='+', color='g', label='tau - Tyler')
plt.loglog(list_n_points, tau_errors_rgd, marker='.', color='r', label='tau - Riemannian')
plt.legend()
plt.xlabel('Number of data points')
plt.ylabel('MSE on tau')
plt.grid(b=True, which='both')
plt.savefig(path+'_tau.png')
tikzplotlib.save(path+'_tau.tex')
plt.clf()

plt.loglog(list_n_points, sigma_errors_g, marker='*', color='b', label='sigma - Gaussian')
plt.loglog(list_n_points, sigma_errors_t, marker='^', color='k', label='sigma - Tyler - location known')
plt.loglog(list_n_points, sigma_errors_t2, marker='+', color='g', label='sigma - Tyler')
plt.loglog(list_n_points, sigma_errors_rgd, marker='.', color='r', label='sigma - Riemannian')
plt.legend()
plt.xlabel('Number of data points')
plt.ylabel('MSE on sigma')
plt.grid(b=True, which='both')
plt.savefig(path+'_sigma.png')
tikzplotlib.save(path+'_sigma.tex')
plt.clf()
