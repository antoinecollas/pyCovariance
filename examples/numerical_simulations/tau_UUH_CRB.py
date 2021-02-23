import autograd.numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from pyCovariance import monte_carlo

from pyCovariance.evaluation import create_directory

from pyCovariance.features import\
        subspace_SCM,\
        subspace_tau_UUH,\
        subspace_tau_UUH_RGD,\
        tau_UUH,\
        tau_UUH_RGD

from pyCovariance.generation_data import\
        generate_complex_stiefel,\
        generate_textures,\
        sample_complex_tau_UUH_distribution


def main(nb_points, n_MC, verbose=True):
    matplotlib.use('Agg')

    if verbose:
        print('################ tau_UUH model CRB ################')

    seed = 0
    np.random.seed(seed)
    if verbose:
        print('seed:', seed)

    # path to save plot
    folder = create_directory('numerical_simulations')

    if verbose:
        print('####### SIMU ON U #######')
    path = os.path.join(folder, 'tau_UUH_CRB_U')

    # parameters simu U
    p = 15
    k = 3
    N_max = int(1e4)
    alpha = 1
    mu = 0.01

    # uncomment to plot texture cdf
    # import scipy.stats as stats
    # x = np.linspace(0, 100, int(1e4))
    # mu = 0.01
    # y = stats.gamma.cdf(x, a=mu, scale=1/mu)
    # plt.plot(x, y, label=str(mu))
    # mu = 0.1
    # y = stats.gamma.cdf(x, a=mu, scale=1/mu)
    # plt.plot(x, y, label=str(mu))
    # mu = 1
    # y = stats.gamma.cdf(x, a=mu, scale=1/mu)
    # plt.plot(x, y, label=str(mu))
    # mu = 10
    # y = stats.gamma.cdf(x, a=mu, scale=1/mu)
    # plt.plot(x, y, label=str(mu))
    # plt.legend()
    # plt.show()

    # generate tau and U
    tau_full = alpha*generate_textures(N_max, mu)
    U = generate_complex_stiefel(p, k)

    # 3 methods of estimation of U
    features_list = [
        subspace_SCM(k),
        subspace_tau_UUH(k, estimate_sigma=False),
        subspace_tau_UUH_RGD(k, estimate_sigma=False)
    ]

    # simu
    list_n_points = np.geomspace(p, N_max, num=nb_points).astype(int)
    mean_errors = np.zeros((len(features_list), nb_points))

    iterator = list_n_points
    if verbose:
        iterator = tqdm(iterator)
    for i, n in enumerate(iterator):
        def sample_fct():
            return sample_complex_tau_UUH_distribution(tau_full[:n], U)

        mean_errors[:, i] = monte_carlo(
            U,
            sample_fct,
            features_list,
            n_MC,
            n_jobs=-1,
            verbose=False
        )

    c_tau = np.mean((tau_full*tau_full)/(1+tau_full))
    bound = ((p-k)*k) / (list_n_points*c_tau)

    # plot
    plt.loglog(list_n_points, mean_errors[0], label='SCM', marker='+')
    plt.loglog(list_n_points, mean_errors[1], label='BCD', marker='x')
    plt.loglog(list_n_points, mean_errors[2], label='RO', marker='2')
    plt.loglog(list_n_points, bound, label='CRB', linestyle='dashed')
    plt.legend()
    plt.xlabel('Number of points')
    plt.ylabel('MSE on U')
    plt.grid(b=True, which='both')

    plt.savefig(path)
    plt.close('all')

    if verbose:
        print()
        print('####### SIMU ON tau #######')
    path = os.path.join(folder, 'tau_UUH_CRB_tau')

    # parameters simu tau
    p = 15
    k = 3
    N = int(1e4)
    alpha_max = int(1e9)
    mu = 1

    # generate tau and U
    tau = generate_textures(N, mu)
    assert tau.shape == (N, 1)
    U = generate_complex_stiefel(p, k)
    assert U.shape == (p, k)

    # features
    features_list = [
        tau_UUH(k, estimate_sigma=False),
        tau_UUH_RGD(k, estimate_sigma=False)
    ]

    # simu
    list_alpha = np.geomspace(1, alpha_max, num=nb_points)
    mean_errors = np.zeros((len(features_list), len(list_alpha)))

    iterator = list_alpha
    if verbose:
        iterator = tqdm(iterator)

    for i, alpha in enumerate(iterator):
        def sample_fct():
            return sample_complex_tau_UUH_distribution(alpha*tau, U)

        mean_errors[:, i] = monte_carlo(
            [alpha*tau, U],
            sample_fct,
            features_list,
            n_MC,
            n_jobs=-1,
            verbose=False
        )[:, 1]

    tau = list_alpha*tau
    tmp = ((1+tau) / tau)**2
    c = np.sum(tmp, axis=0)
    bound = 1/k * c
    bound_2 = N / k * np.ones(len(list_alpha))

    # plot
    plt.loglog(list_alpha, mean_errors[0], label='BCD', marker='x')
    plt.loglog(list_alpha, mean_errors[1], label='RO', marker='2')
    plt.loglog(list_alpha, bound, label='CRB', linestyle='dashed')
    plt.loglog(list_alpha, bound_2, label='N/k', linestyle='dashed')
    plt.legend()
    plt.xlabel('alpha')
    plt.ylabel('MSE on tau')
    plt.grid(b=True, which='both')

    plt.savefig(path)
    plt.close('all')


if __name__ == '__main__':
    main(nb_points=10, n_MC=1000)
