import matplotlib.pyplot as plt
import autograd.numpy as np

from pyCovariance.generation_data import\
        generate_stiefel,\
        generate_textures_lognormal_dist,\
        sample_tau_UUH_distribution
from pyCovariance.features.low_rank_models import\
        estimate_tau_UUH_RO


def main(plot=True):
    plt.style.use('seaborn-white')

    N = int(1e3)
    alpha = 5

    U = generate_stiefel(p=3, k=3)
    U_perp = U[:, 2]
    U = U[:, :2]
    tau = generate_textures_lognormal_dist(N)
    tau = tau * alpha

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # scatter plot of points
    X = sample_tau_UUH_distribution(tau, U)
    ax.scatter(X[0, :], X[1, :], X[2, :], alpha=1)

    # estimation of the subspace
    _, U_est, _ = estimate_tau_UUH_RO(X, k=2)
    U_est_perp = np.cross(U_est[:, 0], U_est[:, 1])

    # plot of subspaces
    min_x = np.min(X[0, :])
    max_x = np.max(X[0, :])
    min_y = np.min(X[1, :])
    max_y = np.max(X[1, :])
    interval_x = np.linspace(min_x-1, max_x+1, num=10)
    interval_y = np.linspace(min_y-1, max_y+1, num=10)
    x, y = np.meshgrid(interval_x, interval_y)
    z = (- U_perp[0]*x - U_perp[1]*y) / U_perp[2]
    z_est = (- U_est_perp[0]*x - U_est_perp[1]*y) / U_est_perp[2]
    ax.plot_surface(x, y, z, alpha=0.4, color='orange', linewidth=0)
    ax.plot_surface(x, y, z_est, alpha=0.4, color='red', linewidth=0)

    if plot:
        plt.show()
    else:
        plt.clf()


if __name__ == '__main__':
    main()
