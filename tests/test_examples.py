from pyCovariance.features import\
        pixel_euclidean,\
        tau_UUH

from examples.hyperspectral import\
        k_means_window_size_nb_bands as hyper_k_means
from examples.hyperspectral.hyperspectral_functions import Dataset

def test_hyperspectral():
    dataset_name = 'Indian_Pines'
    dataset = Dataset(dataset_name)
    p = dataset.dimension

    pairs_w_k = [(3, 4), (5, 4)]

    features_list = list()
    for w, k in pairs_w_k:
        features_list.append([
            'sklearn',
            #pixel_euclidean(k),
            tau_UUH(p, k, w*w)
        ])

    hyper_k_means.main(
        dataset=dataset,
        crop_image=True,
        enable_multi=False,
        pairs_window_size_nb_bands=pairs_w_k,
        mask=True,
        features_list=features_list,
        nb_init=1,
        nb_iter_max=1
    )

