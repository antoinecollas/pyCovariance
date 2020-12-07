from pyCovariance.datasets.hyperspectral import Dataset
from pyCovariance.features import tau_UUH

from examples.hyperspectral import k_means


def test_hyperspectral_indian_pines():
    dataset_name = 'Indian_Pines'
    dataset = Dataset(dataset_name)
    p = dataset.dimension

    pairs_w_k = [(3, 4), (5, 4)]

    features_list = list()
    for w, k in pairs_w_k:
        features_list.append([
            'sklearn',
            tau_UUH(w*w, p, k)
        ])

    k_means.main(
        dataset=dataset,
        crop_image=True,
        nb_threads=1,
        pairs_window_size_nb_bands=pairs_w_k,
        mask=True,
        features_list=features_list,
        nb_init=1,
        nb_iter_max=1
    )


def test_hyperspectral_pavia():
    dataset_name = 'Pavia'
    dataset = Dataset(dataset_name)

    pairs_w_k = [(3, 4)]

    features_list = list()
    for w, k in pairs_w_k:
        features_list.append([
            'sklearn',
        ])

    k_means.main(
        dataset=dataset,
        crop_image=True,
        nb_threads=1,
        pairs_window_size_nb_bands=pairs_w_k,
        mask=True,
        features_list=features_list,
        nb_init=1,
        nb_iter_max=1
    )
