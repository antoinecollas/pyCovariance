from datetime import datetime
import os
import sys

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(1, temp)

from clustering_SAR.features import Covariance, CovarianceEuclidean, CovarianceTexture, Intensity, LocationCovarianceEuclidean, MeanPixelEuclidean, PixelEuclidean
from examples.hyperspectral.hyperspectral_functions import K_means_hyperspectral_image, Dataset, evaluate_and_save_clustering, HyperparametersKMeans

#######################################################
# BEGINNING OF HYPERPARAMETERS
#######################################################

hyperparams = HyperparametersKMeans(
    crop_image = False,
    enable_multi = True,
    pca = True,
    nb_bands_to_select = 10,
    mask = True,
    windows_size = 5,
    features = None,
    nb_init = 20,
    nb_iter_max = 100,
    eps = 1e-3
)

features_list = [
    PixelEuclidean(),
    MeanPixelEuclidean(),
    Intensity(),
    CovarianceEuclidean(),
    Covariance(),
    CovarianceTexture(
        p=hyperparams.nb_bands_to_select,
        N=hyperparams.windows_shape[0]*hyperparams.windows_shape[1]
    )
]

dataset_name = 'Indian_Pines'

#######################################################
# END OF HYPERPARAMETERS
#######################################################

print('################################################')
print('Dataset', dataset_name)
print('################################################')
print()

# folder path to save files
date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
folder = os.path.join('results', dataset_name, date_str)

for i, features in enumerate(features_list):
    hyperparams.features = features
    print()
    print('Features:', str(hyperparams.features))
    C = K_means_hyperspectral_image(dataset_name, hyperparams)
    print()
    evaluate_and_save_clustering(C, dataset_name, hyperparams, folder, str(i))

