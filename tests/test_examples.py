from pyCovariance.datasets.hyperspectral import Dataset
from pyCovariance.features import covariance_texture,\
        center_intensity_euclidean,\
        center_euclidean

from examples.hyperspectral import demo, k_means
from examples.numerical_simulations import\
        tau_UUH_CRB,\
        location_covariance_texture as location_covariance_texture_MSE
from examples.numerical_simulations.plot import\
        subspace,\
        tyler_type_estimator


def test_hyperspectral_demo_K_means():
    demo.main(
        plot=False,
        crop_image=True,
        n_init=1,
        n_iter_max=1,
        verbose=False
    )


def test_hyperspectral_indian_pines():
    dataset_name = 'Indian_Pines'
    dataset = Dataset(dataset_name)

    pairs_w_k = [(5, 5)]

    features_list = list()
    for w, k in pairs_w_k:
        features_list.append([
            center_intensity_euclidean(),
            'sklearn',
            center_euclidean(k),
            covariance_texture(k, w*w),
        ])

    k_means.main(
        dataset=dataset,
        crop_image=True,
        nb_threads=1,
        pairs_window_size_nb_bands=pairs_w_k,
        mask=True,
        features_list=features_list,
        nb_init=1,
        nb_iter_max=1,
        verbose=False
    )

# Pavia and Salilas are commented because data are too heavy
# to run on github actions
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
#         nb_threads=1,
#         pairs_window_size_nb_bands=pairs_w_k,
#         mask=True,
#         features_list=features_list,
#         nb_init=1,
#         nb_iter_max=1,
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
#         nb_threads=1,
#         pairs_window_size_nb_bands=pairs_w_k,
#         mask=True,
#         features_list=features_list,
#         nb_init=1,
#         nb_iter_max=1,
#         verbose=False
#     )


def test_plot_subspace():
    subspace.main(plot=False)


def test_plot_tyler_type_estimator():
    tyler_type_estimator.main(plot=False)


def test_tau_UUH_CRB():
    tau_UUH_CRB.main(
        nb_points=2,
        nb_MC=2,
        verbose=False
    )


def test_location_covariance_texture_MSE():
    location_covariance_texture_MSE.main(
        nb_points=2,
        nb_MC=2,
        iter_max_RGD=100,
        verbose=False
    )
