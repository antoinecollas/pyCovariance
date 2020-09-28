import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

from pyCovariance.features.covariance import distance_covariance_Riemannian
from pyCovariance.features.location_covariance_texture import gradient_descent_location_covariance_texture, tyler_estimator_location_covariance_normalisedet
from pyCovariance.generation_data import generate_covariance, generate_texture, generate_Toeplitz, sample_compound
from pyCovariance.vectorization import unvech, vech


nb_MC = 100
p = 5
N_max = 1000
nb_points = 5

mu = np.random.randn(p, 1) + 1j*np.random.randn(p, 1)
tau_full = generate_texture(N_max)
sigma = generate_covariance(p)
# sigma = generate_Toeplitz(0.1, p)
sigma = (1/np.linalg.det(sigma))**(1/p) * sigma

assert np.abs(np.linalg.det(sigma)-1) < 1e-5

list_n_points = np.geomspace(10*p, N_max, num=nb_points, dtype=np.int)

mu_errors_t = list()
sigma_errors_t = list()

mu_errors_g = list()
sigma_errors_g = list()

for n in tqdm(list_n_points):
    mu_error_t = 0
    sigma_error_t = 0
    
    mu_error_g = 0
    sigma_error_g = 0
    
    tau = tau_full[:n]
    
    for i in tqdm(range(nb_MC)):
        X = sample_compound(tau, sigma) + mu
        #print('np.linalg.norm(X-mu, axis=0)=', np.linalg.norm(X-mu, axis=0))

        # Tyler
        mu_est, tau_est, sigma_est, _, _ = tyler_estimator_location_covariance_normalisedet(X)
        mu_error_t += np.linalg.norm(mu-mu_est)**2
        sigma_error_t += distance_covariance_Riemannian(vech(sigma_est), vech(sigma))**2
        
        # gradient descent 
        mu_est, tau_est, sigma_est = gradient_descent_location_covariance_texture(X, autodiff=False)
        mu_error_g += np.linalg.norm(mu-mu_est)**2
        sigma_error_g += distance_covariance_Riemannian(vech(sigma_est), vech(sigma))**2

    # Tyler
    mu_errors_t.append(mu_error_t/nb_MC)
    sigma_errors_t.append(sigma_error_t/nb_MC)

    # gradient descent 
    mu_errors_g.append(mu_error_g/nb_MC)
    sigma_errors_g.append(sigma_error_g/nb_MC)

plt.loglog(list_n_points, mu_errors_t, label='mu - Tyler')
plt.loglog(list_n_points, mu_errors_g, label='mu - Riemannian')
plt.loglog(list_n_points, sigma_errors_t, label='sigma - Tyler')
plt.loglog(list_n_points, sigma_errors_g, label='sigma - Riemannien')
plt.legend()
plt.xlabel('Nombre de points')
plt.ylabel('Erreur d\'estimation')
plt.savefig('tyler.png')

