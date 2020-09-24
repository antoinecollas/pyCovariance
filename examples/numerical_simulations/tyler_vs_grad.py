import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(1, temp)

from clustering_SAR.covariance_clustering_functions import distance_covariance_Riemannian
from clustering_SAR.estimation import gradient_descent_location_covariance_texture, tyler_estimator_location_covariance_normalisedet
from clustering_SAR.generation_data import generate_covariance, generate_texture, generate_Toeplitz, sample_compound
from clustering_SAR.generic_functions import unvech, vech


nb_MC = 10
p = 3
N_max = 1000*p

mu = np.random.randn(p, 1) + 1j*np.random.randn(p, 1)
tau_full = generate_texture(N_max)
sigma = generate_covariance(p)
# sigma = generate_Toeplitz(0.1, p)
sigma = (1/np.linalg.det(sigma))**(1/p) * sigma

assert np.abs(np.linalg.det(sigma)-1) < 1e-5

list_n_points = np.geomspace(10*p, N_max, num=3, dtype=np.int)

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
        mu_est, tau_est, sigma_est = gradient_descent_location_covariance_texture(X, autodiff=True)
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
plt.show()


