import autograd.numpy as np
from autograd.numpy import random as rnd
from functools import partial
import matplotlib.pyplot as plt
import os

from pyCovariance import monte_carlo

from pyCovariance.evaluation import create_directory

from pyCovariance.features import\
        covariance_texture,\
        location_covariance_texture_Gaussian,\
        location_covariance_texture_Tyler,\
        location_covariance_texture_RGD

from pyCovariance.generation_data import\
        generate_complex_covariance,\
        generate_textures,\
        sample_complex_compound_distribution


def main(
    nb_points,
    nb_MC,
    iter_max_RGD,
    verbose=True
):
    if verbose:
        print('################ Compound Gaussian estimation ################')

    seed = 0
    rnd.seed(seed)
    if verbose:
        print('seed:', seed)

    # path to save plot
    folder = os.path.join('numerical_simulations')
    folder = create_directory(folder)

    # parameters simu
    p = 10
    N_min = 2*p
    N_max = 500

    # generate mu, tau, sigma
    mu = rnd.randn(p, 1)
    tau_full = generate_textures(N_max)
    sigma = generate_complex_covariance(p, unit_det=True)

    # features
    features_list = [location_covariance_texture_Gaussian,
                     location_covariance_texture_Tyler,
                     partial(location_covariance_texture_RGD,
                             iter_max=iter_max_RGD)]

    # simu
    list_n_points = np.geomspace(N_min, N_max, num=nb_points).astype(int)

    # 4 distances
    mean_errors = np.zeros((4, len(features_list)+1, nb_points))

    for i, N in enumerate(list_n_points):
        if verbose:
            print('##### N=' + str(N) + ' #####')
        t = tau_full[:N]

        # location + covariance + texture estimators
        def sample_fct():
            return sample_complex_compound_distribution(t, sigma) + mu
        features = [f(p, N) for f in features_list]
        true_parameters = [mu, sigma, t]
        mean_errors[:, :-1, i] = monte_carlo(
            true_parameters,
            sample_fct,
            features,
            nb_MC,
            verbose
        )

        # Tyler estimator
        def sample_fct():
            return sample_complex_compound_distribution(t, sigma)
        features = [covariance_texture(p, N)]
        true_parameters = [sigma, t]
        mean_errors[[0, 2, 3], -1, i] = monte_carlo(
            true_parameters,
            sample_fct,
            features,
            nb_MC,
            verbose
        ).reshape(-1)

    # plot MSE of location estimation
    plt.figure()
    plt.loglog(list_n_points, mean_errors[1][0], label='Gaussian', marker='+')
    plt.loglog(list_n_points, mean_errors[1][1], label='Tyler', marker='x')
    plt.loglog(list_n_points, mean_errors[1][2], label='RGD', marker='2')
    plt.legend()
    plt.xlabel('Number of points')
    plt.ylabel('MSE of location estimation')
    plt.grid(b=True, which='both')

    path_temp = os.path.join(folder, 'location_covariance_texture_mu')
    plt.savefig(path_temp)

    # plot MSE of scatter matrix estimation
    plt.figure()
    plt.loglog(list_n_points, mean_errors[2][0], label='Gaussian', marker='+')
    plt.loglog(list_n_points, mean_errors[2][1], label='Tyler', marker='x')
    plt.loglog(list_n_points, mean_errors[2][2], label='RGD', marker='2')
    plt.loglog(list_n_points, mean_errors[2][3],
               label='Tyler - location known', marker='.')
    plt.legend()
    plt.xlabel('Number of points')
    plt.ylabel('MSE on scatter matrix estimation')
    plt.grid(b=True, which='both')

    path_temp = os.path.join(folder, 'location_covariance_texture_scatter')
    plt.savefig(path_temp)


if __name__ == '__main__':
    main(nb_points=5, nb_MC=200, iter_max_RGD=3*int(1e4))
