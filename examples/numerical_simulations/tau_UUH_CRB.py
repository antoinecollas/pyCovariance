import autograd.numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import tikzplotlib

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
        generate_textures_lognormal_dist,\
        sample_complex_tau_UUH_distribution


def main(n_points, n_MC, p, k, N_max_simu_U, N_simu_tau, verbose=True):
    matplotlib.use('Agg')

    # uncomment to plot cdf
    # from scipy import stats
    # x = np.linspace(0, 10, num=int(1e5))

    # nu = 0.1
    # y = stats.gamma.cdf(x, a=nu, scale=1/nu)
    # plt.plot(x, y, label='gamma(0.1, 10)')

    # nu = 0.01
    # y = stats.gamma.cdf(x, a=nu, scale=1/nu)
    # plt.plot(x, y, label='gamma(0.01, 100)')

    # variance = 2
    # s = np.sqrt(variance)
    # mu = -(s**2)/2
    # y = stats.lognorm.cdf(x, scale=np.exp(mu), s=s)
    # plt.plot(x, y, label='lognorm(var=2)')

    # variance = 5
    # s = np.sqrt(variance)
    # mu = -(s**2)/2
    # y = stats.lognorm.cdf(x, scale=np.exp(mu), s=s)
    # plt.plot(x, y, label='lognorm(var=5)')

    # plt.legend()
    # plt.show()

    # import sys
    # sys.exit(0)

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
    N_max = N_max_simu_U
    SNR_list = [1, 10]
    var_list = [4, 2]

    for SNR in SNR_list:
        for var in var_list:
            if verbose:
                print('SNR:', SNR, 'var:', var)

            suffix = '_SNR_' + str(SNR) + '_var_' + str(var)
            suffix = suffix.replace('.', '_')
            full_path = path + suffix

            # generate tau and U
            tau_full = SNR*generate_textures_lognormal_dist(N_max, var)
            U = generate_complex_stiefel(p, k)

            # 3 methods of estimation of U
            features_list = [
                subspace_SCM(k),
                subspace_tau_UUH(k, estimate_sigma=False),
                subspace_tau_UUH_RGD(k, estimate_sigma=False)
            ]

            # simu
            list_n_points = np.geomspace(2*p, N_max, num=n_points).astype(int)
            mean_errors = np.zeros((len(features_list), n_points))

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

            plt.savefig(full_path)
            tikzplotlib.save(full_path)

            plt.close('all')

    if verbose:
        print()
        print('####### SIMU ON tau #######')
    path = os.path.join(folder, 'tau_UUH_CRB_tau')

    # parameters simu tau
    N = N_simu_tau
    SNR_min = 100
    SNR_max = int(1e6)
    var_list = [2, 4]

    # generate U
    U = generate_complex_stiefel(p, k)

    # features
    features_list = [
        tau_UUH(k, estimate_sigma=False),
        tau_UUH_RGD(k, estimate_sigma=False)
    ]

    # simu
    list_SNR = np.geomspace(SNR_min, SNR_max, num=n_points)
    mean_errors = np.zeros((len(features_list), len(list_SNR)))

    for var in var_list:
        if verbose:
            print('var:', var)

        suffix = '_var_' + str(var)
        suffix = suffix.replace('.', '_')
        full_path = path + suffix

        tau = generate_textures_lognormal_dist(N, var)

        iterator = list_SNR
        if verbose:
            iterator = tqdm(iterator)

        for i, SNR in enumerate(iterator):
            def sample_fct():
                return sample_complex_tau_UUH_distribution(SNR*tau, U)

            mean_errors[:, i] = monte_carlo(
                [SNR*tau, U],
                sample_fct,
                features_list,
                n_MC,
                n_jobs=-1,
                verbose=False
            )[:, 1]

        tau = list_SNR*tau
        tmp = ((1+tau) / tau)**2
        c = np.sum(tmp, axis=0)
        bound = 1/k * c
        bound_2 = N / k * np.ones(len(list_SNR))

        mean_errors = mean_errors / N
        bound = bound / N
        bound_2 = bound_2 / N

        # plot
        plt.loglog(list_SNR, mean_errors[0], label='BCD', marker='x')
        plt.loglog(list_SNR, mean_errors[1], label='RO', marker='2')
        plt.loglog(list_SNR, bound, label='CRB', linestyle='dashed')
        plt.loglog(list_SNR, bound_2, label='1/k', linestyle='dashed')
        plt.ylim(ymin=3*1e-2, ymax=10)
        plt.legend()
        plt.xlabel('SNR')
        plt.ylabel('MSE on tau')
        plt.grid(b=True, which='both')

        plt.savefig(full_path)
        tikzplotlib.save(full_path)
        plt.close('all')


if __name__ == '__main__':
    main(
        n_points=10,
        n_MC=100,
        p=100,
        k=20,
        N_max_simu_U=int(5*1e4),
        N_simu_tau=int(1e4)
    )
