from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

from pyCovariance.features import Covariance, CovarianceEuclidean, CovarianceTexture, Intensity, LocationCovarianceEuclidean, MeanPixelEuclidean, PixelEuclidean

from hyperspectral_functions import K_means_hyperspectral_image, Dataset, evaluate_and_save_clustering, HyperparametersKMeans


dataset_name = 'Indian_Pines'

# folder path to save files
date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
folder = os.path.join('results', dataset_name, date_str)


# EVALUATION OF PCA

hyperparams = HyperparametersKMeans(
    crop_image = True,
    enable_multi = True,
    pca = None,
    nb_bands_to_select = 2,
    mask = True,
    windows_size = 7,
    features = None,
    nb_init = 1,
    nb_iter_max = 2,
    eps = 1e-3
)

features_list = [
    Intensity(),
    'sklearn',
    PixelEuclidean(),
    MeanPixelEuclidean(),
    CovarianceEuclidean(),
    Covariance(),
    CovarianceTexture(
        p=hyperparams.nb_bands_to_select,
        N=hyperparams.windows_shape[0]*hyperparams.windows_shape[1]
    )
]

for pca in [False, True]:
    hyperparams.pca = pca
    if pca:
        prefix = '_pca'
    else:
        prefix = '_no_pca'

    # K means and evaluations
    mIoUs = list()
    features_str = list()
    for i, features in enumerate(features_list):
        hyperparams.features = features
        print()
        print('Features:', str(hyperparams.features))
        C = K_means_hyperspectral_image(dataset_name, hyperparams)
        print()
        mIoU, _ = evaluate_and_save_clustering(C, dataset_name, hyperparams, folder, str(i)+prefix)
        mIoUs.append(mIoU)
        features_str.append(str(features))

    # Bar plot of performances
    fig, ax = plt.subplots(1)
    ax.bar(features_str, mIoUs, align='center')
    ax.set_ylim(0, 0.5)
    plt.ylabel('mIoU')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.5)
    path = os.path.join('results', dataset_name, date_str, 'perfs'+prefix)
    plt.savefig(path)
