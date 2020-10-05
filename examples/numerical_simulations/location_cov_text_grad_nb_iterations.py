import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pyCovariance.features.location_covariance_texture import estimation_location_covariance_texture_RGD
from pyCovariance.generation_data import generate_covariance, generate_texture, sample_compound


nb_MC = 20
p = 10
N = 50
iter_max = 10000

mu = np.random.randn(p, 1) + 1j*np.random.randn(p, 1)
tau = generate_texture(N)
sigma = generate_covariance(p)
sigma = (1/np.linalg.det(sigma))**(1/p) * sigma

assert np.abs(np.linalg.det(sigma)-1) < 1e-5

f_values_steepest = list()
grad_norm_values_steepest = list()

f_values_conjugate = list()
grad_norm_values_conjugate = list()

for i in tqdm(range(nb_MC)):
    X = sample_compound(tau, sigma) + mu

    _, _, _, log = estimation_location_covariance_texture_RGD(
        X,
        tol=-1,
        iter_max=iter_max,
        autodiff=False,
        solver='steepest'
    )
    f_values_steepest.append(np.array(log['iterations']['f(x)']))
    grad_norm_values_steepest.append(log['iterations']['gradnorm'])

    _, _, _, log = estimation_location_covariance_texture_RGD(
        X,
        tol=-1,
        iter_max=iter_max,
        autodiff=False,
        solver='conjugate'
    )
    f_values_conjugate.append(np.array(log['iterations']['f(x)']))
    grad_norm_values_conjugate.append(log['iterations']['gradnorm'])

min_f_values = np.min(np.concatenate([np.array(f_values_steepest), np.array(f_values_conjugate)]))

f_values_steepest = np.stack(f_values_steepest)
f_values_steepest = np.mean(f_values_steepest, axis=0)
f_values_steepest = f_values_steepest - min_f_values + 1

grad_norm_values_steepest = np.stack(grad_norm_values_steepest)
grad_norm_values_steepest = grad_norm_values_steepest.reshape((grad_norm_values_steepest.shape[0], grad_norm_values_steepest.shape[1]))
grad_norm_values_steepest = np.mean(grad_norm_values_steepest, axis=0)

f_values_conjugate = np.stack(f_values_conjugate)
f_values_conjugate = np.mean(f_values_conjugate, axis=0)
f_values_conjugate = f_values_conjugate - min_f_values + 1

grad_norm_values_conjugate = np.stack(grad_norm_values_conjugate)
grad_norm_values_conjugate = grad_norm_values_conjugate.reshape((grad_norm_values_conjugate.shape[0], grad_norm_values_conjugate.shape[1]))
grad_norm_values_conjugate = np.mean(grad_norm_values_conjugate, axis=0)


list_iter = list(range(1, iter_max+1))

plt.loglog(list_iter, f_values_steepest, label='steepest descent')
plt.loglog(list_iter, f_values_conjugate, label='conjugate gradient')
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Negative log likelihood')
plt.grid(b=True, which='both')
plt.savefig('results/location_cov_text_RGD_likelihood.png')
plt.clf()

plt.loglog(list_iter, grad_norm_values_steepest, label='steepest descent')
plt.loglog(list_iter, grad_norm_values_conjugate, label='conjugate gradient')
plt.xlabel('Number of iterations')
plt.ylabel('Gradient norm')
plt.grid(b=True, which='both')
plt.savefig('results/location_cov_text_RGD_gradient_norm.png')
