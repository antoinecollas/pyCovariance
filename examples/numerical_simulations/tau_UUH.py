import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pyCovariance.features.tau_UUH import distance_grass, estimation_tau_UUH, estimation_tau_UUH_gradient_descent, estimation_tau_UUH_SCM
from pyCovariance.generation_data import generate_stiefel, generate_texture, sample_tau_UUH

nb_MC = 100
p = 15
k = 3
N_max = 1000
nb_points = 10
alpha = 10

tau_full = generate_texture(N_max)
U = generate_stiefel(p, k)

list_n_points = np.geomspace(p, N_max, num=nb_points, dtype=np.int)

U_errors_SCM = list()
U_errors_alternate = list()
U_errors_grad = list()

for n in tqdm(list_n_points):
    U_error_SCM = 0
    U_error_alternate = 0
    U_error_grad = 0
 
    tau = tau_full[:n]

    for i in tqdm(range(nb_MC)):
        X = sample_tau_UUH(tau, U)

        # projected SCM
        U_est, _ = estimation_tau_UUH_SCM(X, k)
        U_error_SCM += distance_grass(U_est, U)**2

        # alternate tau UUH
        U_est, _, _, _ = estimation_tau_UUH(X, k)
        U_error_alternate += distance_grass(U_est, U)**2

        # gradient descent tau UUH
        U_est, _ = estimation_tau_UUH_gradient_descent(X, k, autodiff=False)
        U_error_grad += distance_grass(U_est, U)**2

    U_errors_SCM.append(U_error_SCM/nb_MC)
    U_errors_alternate.append(U_error_alternate/nb_MC)
    U_errors_grad.append(U_error_grad/nb_MC)

plt.loglog(list_n_points, U_errors_SCM, label='U - projected SCM')
plt.loglog(list_n_points, U_errors_alternate, label='U - alternate')
plt.loglog(list_n_points, U_errors_grad, label='U - Riemannian')
plt.legend()
plt.xlabel('Number of points')
plt.ylabel('Estimation error')
plt.grid(b=True, which='both')
plt.savefig('tau_UUH.png')
