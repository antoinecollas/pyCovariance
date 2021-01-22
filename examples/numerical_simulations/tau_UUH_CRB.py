import autograd.numpy as np
import matplotlib.pyplot as plt
import os

from pyCovariance import monte_carlo

from pyCovariance.evaluation import create_directory

from pyCovariance.features import\
        subspace_SCM,\
        subspace_tau_UUH,\
        subspace_tau_UUH_RGD

from pyCovariance.generation_data import\
        generate_complex_stiefel,\
        generate_textures,\
        sample_complex_tau_UUH_distribution


def main(nb_points, nb_MC):
    print('################ tau_UUH model CRB ################')

    seed = 0
    np.random.seed(seed)
    print('seed:', seed)

    # path to save plot
    folder = create_directory('numerical_simulations')
    path = os.path.join(folder, 'tau_UUH_CRB')

    # parameters simu
    p = 15
    k = 3
    N_max = 1000
    alpha = 10

    # generate tau and U
    tau_full = alpha*generate_textures(N_max)
    U = generate_complex_stiefel(p, k)

    # 3 methods of estimation of U
    features_list = [subspace_SCM(p, k),
                     subspace_tau_UUH(p, k),
                     subspace_tau_UUH_RGD(p, k)]

    # simu
    list_n_points = np.geomspace(p, N_max, num=nb_points).astype(int)
    mean_errors = np.zeros((len(features_list), nb_points))

    for i, n in enumerate(list_n_points):
        def sample_fct():
            return sample_complex_tau_UUH_distribution(tau_full[:n], U)
        mean_errors[:, i] = monte_carlo(U, sample_fct, features_list, nb_MC)

    c_tau = np.mean((tau_full*tau_full)/(1+tau_full))
    bound = (((p-k)*k)/(list_n_points*c_tau))

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


if __name__ == '__main__':
    main(nb_points=10, nb_MC=500)
