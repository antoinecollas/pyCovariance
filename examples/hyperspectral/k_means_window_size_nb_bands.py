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


# EVALUATION ACCORDING TO WINDOW_SIZE AND NB_BANDS

hyperparams = HyperparametersKMeans(
    crop_image = False,
    enable_multi = True,
    pca = True,
    nb_bands_to_select = None,
    mask = True,
    windows_size = None,
    features = None,
    nb_init = 10,
    nb_iter_max = 100,
    eps = 1e-3
)

pairs_w_p = [(3, 4), (5, 4), (5, 10), (7, 4), (7, 10), (7, 20), (9, 4), (9, 10), (9, 20), (9, 40)]

for w, p in pairs_w_p:
    print('w=', w, 'p=', p)

    hyperparams.nb_bands_to_select = p
    hyperparams.windows_size = w

    features_list = [
        Intensity(),
        'sklearn',
        PixelEuclidean(),
        MeanPixelEuclidean(),
        CovarianceEuclidean(),
        Covariance(),
        CovarianceTexture(p=p, N=w*w)
    ]

    prefix = 'w' + str(w) + '_p' + str(p)

    # K means and evaluations
    mIoUs = list()
    OAs = list()
    features_str = list()
    for i, features in enumerate(features_list):
        hyperparams.features = features
        print()
        print('Features:', str(hyperparams.features))
        C = K_means_hyperspectral_image(dataset_name, hyperparams)
        print()
        mIoU, OA = evaluate_and_save_clustering(C, dataset_name, hyperparams, folder, str(i) + '_' + prefix)
        mIoUs.append(mIoU)
        OAs.append(OA)
        features_str.append(str(features))

    # Bar plot of mIoUs
    fig, ax = plt.subplots(1)
    ax.bar(features_str, mIoUs, align='center')
    ax.set_ylim(0, 0.5)
    plt.ylabel('mIoU')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.5)
    path = os.path.join('results', dataset_name, date_str, 'mIoU'+prefix)
    plt.savefig(path)

    # Bar plot of OAs
    fig, ax = plt.subplots(1)
    ax.bar(features_str, OAs, align='center')
    ax.set_ylim(0, 1)
    plt.ylabel('OA')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.5)
    path = os.path.join('results', dataset_name, date_str, 'OA'+prefix)
    plt.savefig(path)
