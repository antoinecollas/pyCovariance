import autograd.numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import tikzplotlib

from pyCovariance import monte_carlo

from pyCovariance.evaluation import create_directory

from pyCovariance.features import subspace_tau_UUH_RO

from pyCovariance.generation_data import\
        generate_complex_stiefel,\
        generate_textures_lognormal_dist,\
        sample_complex_tau_UUH_distribution


def main(n_points, n_MC, p, k, N_min, N_max,
         SNR_list, var_list, batch_size, n_epochs, n_jobs, verbose=True):

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
    path = os.path.join(folder, 'tau_UUH_SGD_CRB_U')

    # parameters simu U
    N_ESTIMATORS = 1

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

            # simu
            list_n_points = np.geomspace(
                N_min, N_max, num=n_points).astype(int)
            mean_errors = np.zeros((N_ESTIMATORS, n_points))

            iterator = list_n_points
            if verbose:
                iterator = tqdm(iterator)
            for i, n in enumerate(iterator):
                def sample_fct():
                    return sample_complex_tau_UUH_distribution(tau_full[:n], U)

                iter_max = int(n_epochs * (n // batch_size))
                if verbose:
                    verbosity = 2
                    print('iter_max=', iter_max)
                else:
                    verbosity = 0
                features_list = [
                    subspace_tau_UUH_RO(
                        k, estimate_sigma=False,
                        solver='SGD',
                        information_geometry=True,
                        batch_size=batch_size,
                        iter_max=iter_max,
                        verbosity=verbosity
                    )
                ]
                assert N_ESTIMATORS == len(features_list)

                mean_errors[:, i] = monte_carlo(
                    U,
                    sample_fct,
                    features_list,
                    n_MC,
                    n_jobs=n_jobs,
                    verbose=False
                )

            c_tau = np.mean((tau_full*tau_full)/(1+tau_full))
            bound = ((p-k)*k) / (list_n_points*c_tau)

            # plot
            plt.loglog(list_n_points, mean_errors[0],
                       label='RO-SGD', marker='2')
            plt.loglog(list_n_points, bound,
                       label='CRB', linestyle='dashed', marker='+')
            plt.legend()
            plt.xlabel('Number of points')
            plt.ylabel('MSE on U')
            plt.grid(b=True, which='both')

            plt.savefig(full_path)
            tikzplotlib.save(full_path)

            plt.close('all')


if __name__ == '__main__':
    main(
        n_points=5,
        n_MC=100,
        p=int(1e4),
        k=int(1e1),
        N_min=int(1e3),
        N_max=int(1e4),
        SNR_list=[int(1e3)],
        var_list=[2],
        batch_size=150,
        n_epochs=500,
        n_jobs=10,
        verbose=True
    )
