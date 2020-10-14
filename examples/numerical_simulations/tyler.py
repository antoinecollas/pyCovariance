import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

from pyCovariance.features.covariance import distance_covariance_Riemannian
from pyCovariance.features.covariance_texture import tyler_estimator_covariance_normalisedet
from pyCovariance.generation_data import generate_covariance, generate_texture, sample_compound
from pyCovariance.vectorization import unvech, vech


nb_MC = 100
p = 10
N_max = 10000
nb_points = 5
tol = 1e-10
iter_max = 10000

tau_full = generate_texture(N_max)
sigma = generate_covariance(p)
# sigma = generate_Toeplitz(0.1, p)
sigma = (1/np.linalg.det(sigma))**(1/p) * sigma

assert np.abs(np.linalg.det(sigma)-1) < 1e-5

list_n_points = np.geomspace(2*p, N_max, num=nb_points, dtype=np.int)
print(list_n_points)

# Gaussian
sigma_errors_g = list()

# Tyler
sigma_errors = list()

for n in tqdm(list_n_points):
    sigma_error_g = 0
    sigma_error_o = 0
    sigma_error = 0

    tau = tau_full[:n]

    for i in tqdm(range(nb_MC)):
        X = sample_compound(tau, sigma)

        # Gaussian
        sigma_g = (1/n)*X@X.conj().T
        tau_g = np.real(np.linalg.det(sigma_g)**(1/p))*np.ones((n, 1))
        sigma_g = sigma_g/np.real(np.linalg.det(sigma_g)**(1/p))
        sigma_error_g += distance_covariance_Riemannian(vech(sigma_g), vech(sigma))**2

        # Tyler
        tau_est, sigma_est, _, _ = tyler_estimator_covariance_normalisedet(
            X,
            init=None,
            tol=tol,
            iter_max=iter_max
        )
        sigma_error += distance_covariance_Riemannian(vech(sigma_est), vech(sigma))**2

    # Gaussian
    sigma_errors_g.append(sigma_error_g/nb_MC)

    # Tyler
    sigma_errors.append(sigma_error/nb_MC)


plt.loglog(list_n_points, sigma_errors_g, marker='.', label='sigma - Gaussian')
plt.loglog(list_n_points, sigma_errors, marker='^', label='sigma - Tyler')
plt.loglog([list_n_points[0], list_n_points[-1]], [(p**2-1)/list_n_points[0], (p**2-1)/list_n_points[-1]], marker='^', label='iCRB')

plt.legend()
plt.xlabel('Nombre de points')
plt.ylabel('Erreur d\'estimation')
plt.grid(b=True, which='both')
plt.savefig('results/tyler.png')
