from autograd.numpy import random as rnd
from pyCovariance.datasets.hyperspectral import Dataset
from pyCovariance.features import covariance_texture,\
        center_intensity_euclidean,\
        center_euclidean

from examples.hyperspectral import demo, k_means, metric_weights, stats
from examples.numerical_simulations import\
        tau_UUH_CRB,\
        location_covariance_texture as location_covariance_texture_MSE
from examples.numerical_simulations.plot import\
        subspace,\
        tyler_type_estimator


def test_hyperspectral_demo_K_means():
    rnd.seed(123)

    demo.main(
        plot=False,
        crop_image=True,
        n_init=1,
        max_iter=1,
        verbose=False
    )


def test_hyperspectral_indian_pines():
    rnd.seed(123)

    dataset_name = 'Indian_Pines'
    dataset = Dataset(dataset_name)

    pairs_w_k = [(5, 5)]

    features_list = list()
    for w, k in pairs_w_k:
        features_list.append([
            center_intensity_euclidean(),
            'sklearn',
            center_euclidean(),
            covariance_texture(),
        ])

    k_means.main(
        dataset=dataset,
        crop_image=True,
        border_size=4,
        pairs_window_size_n_bands=pairs_w_k,
        mask=True,
        features_list=features_list,
        n_experiments=1,
        n_init=1,
        max_iter=1,
        n_jobs=1,
        export_tex=False,
        verbose=False
    )


# def test_hyperspectral_pavia():
#     dataset_name = 'Pavia'
#     dataset = Dataset(dataset_name)
#
#     pairs_w_k = [(3, 4)]
#
#     features_list = list()
#     for w, k in pairs_w_k:
#         features_list.append([
#             'sklearn',
#         ])
#
#     k_means.main(
#         dataset=dataset,
#         crop_image=True,
#         n_jobs=1,
#         pairs_window_size_n_bands=pairs_w_k,
#         mask=True,
#         features_list=features_list,
#         n_init=1,
#         max_iter=1,
#         verbose=False
#     )
#
#
# def test_hyperspectral_salinas():
#     dataset_name = 'Salinas'
#     dataset = Dataset(dataset_name)
#
#     pairs_w_k = [(3, 4)]
#
#     features_list = list()
#     for w, k in pairs_w_k:
#         features_list.append([
#             'sklearn',
#         ])
#
#     k_means.main(
#         dataset=dataset,
#         crop_image=True,
#         n_jobs=1,
#         pairs_window_size_n_bands=pairs_w_k,
#         mask=True,
#         features_list=features_list,
#         n_init=1,
#         max_iter=1,
#         verbose=False
#     )


def test_hyperspectral_stats_indian_pines():
    rnd.seed(123)

    dataset_name = 'Indian_Pines'
    dataset = Dataset(dataset_name)

    stats.main(
        dataset=dataset,
        border_size=4,
        mask=True,
        export_tex=False,
        verbose=False
    )


def test_hyperspectral_metric_weights():
    rnd.seed(123)

    dataset_name = 'Indian_Pines'
    dataset = Dataset(dataset_name)

    metric_weights.main(
        dataset=dataset,
        crop_image=True,
        border_size=4,
        window_size=5,
        k=5,
        mask=True,
        estimate_sigma=True,
        n_jobs=-1,
        verbose=False
    )


def test_plot_subspace():
    rnd.seed(123)

    subspace.main(plot=False)


def test_plot_tyler_type_estimator():
    rnd.seed(123)

    tyler_type_estimator.main(plot=False)


def test_tau_UUH_CRB():
    rnd.seed(123)

    tau_UUH_CRB.main(
        n_points=2,
        n_MC=2,
        p=3,
        k=2,
        N_max_simu_U=100,
        N_simu_tau=100,
        verbose=False
    )


def test_location_covariance_texture_MSE():
    rnd.seed(123)

    location_covariance_texture_MSE.main(
        nb_points=2,
        n_MC=2,
        iter_max_RGD=100,
        verbose=False
    )
