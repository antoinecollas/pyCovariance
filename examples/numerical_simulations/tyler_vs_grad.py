import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(1, temp)

from clustering_SAR.covariance_clustering_functions import distance_covariance_Riemannian
from clustering_SAR.estimation import tyler_estimator_location_covariance_normalisedet
from clustering_SAR.generation_data import generate_covariance, generate_texture, generate_Toeplitz, sample_compound
from clustering_SAR.generic_functions import unvech, vech


nb_MC = 1000
N_max = 1000
p = 3
mu = np.random.randn(p, 1) + 1j*np.random.randn(p, 1)
tau_full = generate_texture(N_max)
#sigma = generate_covariance(p)
sigma = generate_Toeplitz(0.1, p)
sigma = (1/np.linalg.det(sigma))**(1/p) * sigma
assert np.abs(np.linalg.det(sigma)-1) < 1e-5

list_n_points = np.geomspace(2*p, N_max, num=10, dtype=np.int)
mu_errors = list()
sigma_errors = list()

for n in list_n_points:
    mu_error = 0
    sigma_error = 0
    tau = tau_full[:n]
    for i in range(nb_MC):
        #print('n=', n, 'i=', i)
        X = sample_compound(tau, sigma) + mu
        #print('np.linalg.norm(X-mu, axis=0)=', np.linalg.norm(X-mu, axis=0))
        mu_est, sigma_est, tau_est, _, _ = tyler_estimator_location_covariance_normalisedet(X)
        mu_error += np.linalg.norm(mu-mu_est)**2
        sigma_error += distance_covariance_Riemannian(vech(sigma_est), vech(sigma))**2
    mu_errors.append(mu_error/nb_MC)
    sigma_errors.append(sigma_error/nb_MC)

plt.loglog(list_n_points, mu_errors, label='Erreur sur mu')
plt.loglog(list_n_points, sigma_errors, label='Erreur sur sigma')
plt.legend()
plt.xlabel('Nombre de points')
plt.ylabel('Erreur d\'estimation')
plt.show()


