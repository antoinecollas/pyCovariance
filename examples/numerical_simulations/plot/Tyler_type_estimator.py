import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from pyCovariance.generation_data import\
        generate_covariance,\
        generate_textures_gamma_dist,\
        sample_compound_distribution
from pyCovariance.features.covariance import\
        compute_scm
from pyCovariance.features.location_covariance_texture import\
        estimate_compound_Gaussian_constrained_scatter


def main(plot=True):
    plt.style.use('seaborn-white')
    transparency = 0.4

    N = 20

    mu = 10*np.random.randn(2, 1)
    cov = generate_covariance(p=2, unit_det=True)
    tau = generate_textures_gamma_dist(N, nu=0.1)
    X = mu + sample_compound_distribution(tau, cov)

    # meshgrid
    nb_points = 300
    x = np.linspace(mu[0]-10, mu[0]+10, nb_points)
    y = np.linspace(mu[1]-10, mu[1]+10, nb_points)
    x, y = np.meshgrid(x, y)
    pos = np.array([x.flatten(), y.flatten()]).T

    plt.figure(1)

    # plot contour of real dist
    dist = multivariate_normal(mu.reshape(-1), cov)
    plt.contour(x, y, dist.pdf(pos).reshape((nb_points, nb_points)),
                colors=['orange'], alpha=transparency)

    # plot contour of dist estimated with mean and scm
    mean = np.mean(X, axis=1, keepdims=True)
    scm = compute_scm(X - mean)
    scm = scm/(np.linalg.det(scm)**(1/2))
    dist = multivariate_normal(mean.reshape(-1), scm)
    plt.contour(x, y, dist.pdf(pos).reshape((nb_points, nb_points)),
                colors=['red'], alpha=transparency)

    # scatter plot of points
    plt.scatter(X[0, :], X[1, :], alpha=1)

    if plot:
        plt.show(block=False)
    else:
        plt.clf()

    plt.figure(2)

    # plot contour of real dist
    dist = multivariate_normal(mu.reshape(-1), cov)
    plt.contour(x, y, dist.pdf(pos).reshape((nb_points, nb_points)),
                colors=['orange'], alpha=transparency)

    # plot contour of dist estimated with Riemannian gradient descent
    mu_est, cov_est, _, _ = estimate_compound_Gaussian_constrained_scatter(
        X,
        iter_max=int(1e3)
    )
    dist = multivariate_normal(mu_est.reshape(-1), cov_est)
    plt.contour(x, y, dist.pdf(pos).reshape((nb_points, nb_points)),
                colors=['red'], alpha=transparency)

    # scatter plot of points
    plt.scatter(X[0, :], X[1, :], alpha=1)

    if plot:
        plt.show()
    else:
        plt.clf()


if __name__ == '__main__':
    main()
