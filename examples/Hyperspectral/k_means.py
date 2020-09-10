from datetime import datetime
import numpy as np
import os
import random
from scipy.io import loadmat
from sklearn.cluster import KMeans
import sys
import time

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(1, temp)

from clustering_SAR.cluster_datacube import K_means_datacube
from clustering_SAR.features import center_vectors_estimation, Covariance, CovarianceEuclidean, CovarianceTexture, Intensity, MeanPixelEuclidean, PixelEuclidean
from clustering_SAR.generic_functions import enable_latex_infigures, pca_and_save_variance, save_figure
from clustering_SAR.evaluation import plot_segmentation, save_segmentation

class Dataset():
    def __init__(self, name):
        self.name = name
        if name == 'Pavia':
            self.path = 'data/Pavia/PaviaU.mat'
            self.key_dict = 'paviaU'
            self.path_gt = 'data/Pavia/PaviaU_gt.mat'
            self.key_dict_gt = 'paviaU_gt'
            self.resolution = [1.3, 1.3] # resolution in meters
        elif name == 'Indian_Pines':
            self.path = 'data/Indian_Pines/Indian_pines_corrected.mat'
            self.key_dict = 'indian_pines_corrected'
            self.path_gt = 'data/Indian_Pines/Indian_pines_gt.mat'
            self.key_dict_gt = 'indian_pines_gt'
            self.resolution = [1.3, 1.3] # resolution in meters
        else:
            raise NotImplementedError

if __name__ == '__main__':

    #######################################################
    #######################################################
    # BEGINNING OF HYPERPARAMETERS
    #######################################################
    #######################################################

    # Dataset
    DATASET_LIST = ['Pavia', 'Indian_Pines']
    CROP_IMAGE = False
    if CROP_IMAGE:
        print()
        print('CROP_IMAGE mode enabled !!!')
        SIZE_CROP = 100
    NB_BANDS_TO_SELECT = 10
    
    # Enable parallel processing (or not)
    ENABLE_MULTI = True
    NUMBER_OF_THREADS_ROWS = os.cpu_count()//2
    NUMBER_OF_THREADS_COLUMNS = 2
    if NUMBER_OF_THREADS_ROWS*NUMBER_OF_THREADS_COLUMNS != os.cpu_count():
        print('ERROR: all cpus are not used ...')
        sys.exit(1)
    NUMBER_OF_THREADS = os.cpu_count() 

    # Apply PCA or select bands randomly
    PCA = False

    # Cluster only the pixels only where there is a ground truth
    MASK = False

    # Window size to compute features
    WINDOWS_SHAPE = (5,5)

    # Features used to cluster the image
    FEATURES_LIST = [PixelEuclidean(), MeanPixelEuclidean(), Intensity(), CovarianceEuclidean(), Covariance(), CovarianceTexture(p=NB_BANDS_TO_SELECT, N=WINDOWS_SHAPE[0]*WINDOWS_SHAPE[1])]

    # K-means parameter
    NUMBER_INIT = 10
    K_MEANS_NB_ITER_MAX = 100
    EPS = 1e-3

    #######################################################
    #######################################################
    # END OF HYPERPARAMETERS
    #######################################################
    #######################################################

    # Folder to save results
    date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
 
    for dataset_name in DATASET_LIST:
        dataset = Dataset(dataset_name)
        FOLDER_RESULTS = os.path.join('results', dataset.name, date_str)
        FOLDER_FIGURES = os.path.join(FOLDER_RESULTS, 'figures')

        print('################################################')
        print('Reading dataset', dataset.name) 
        print('################################################')
        t_beginning = time.time()

        # load image and gt
        image = loadmat(dataset.path)[dataset.key_dict]
        gt = loadmat(dataset.path_gt)[dataset.key_dict_gt]
        number_classes = len(np.unique(gt)) - 1

        # center image globally
        mean = np.mean(image, axis=0)
        image = image - mean
        # check pixels are centered
        assert (np.abs(np.mean(image, axis=0)) < 1e-9).all()

        # pca
        if PCA:
            image = pca_and_save_variance(FOLDER_FIGURES, 'fig_explained_variance', image, NB_BANDS_TO_SELECT)
        else:
            print('Bands are selected randomly.')
            random.seed(2)
            bands = random.sample(list(range(image.shape[2])), k=NB_BANDS_TO_SELECT)
            bands.sort()
            image = image[:, :, bands]

        if CROP_IMAGE:
            center = np.array(image.shape[0:2])//2
            half_height = SIZE_CROP//2
            half_width = SIZE_CROP//2
            image = image[center[0]-half_height:center[0]+half_height, center[1]-half_width:center[1]+half_width]
        n_r, n_c, p = image.shape
        print('image.shape', image.shape)

        # mask
        h = WINDOWS_SHAPE[0]//2
        w = WINDOWS_SHAPE[1]//2
        if MASK:
            mask = (gt != 0)
            mask = mask[h:-h, w:-w]
        else:
            mask = None

        print()
        print('K-means using Sklearn implementation ...') 
        print()

        # We use scikit-learn K-means implementation as a reference
        n_jobs = NUMBER_OF_THREADS_ROWS*NUMBER_OF_THREADS_COLUMNS if ENABLE_MULTI else 1
        sklearn_K_means = KMeans(n_clusters=number_classes, n_init=NUMBER_INIT)
        image_sk = image[h:-h, w:-w].reshape((-1, NB_BANDS_TO_SELECT))
        if MASK:
            image_sk = image_sk[mask.reshape(-1)]
        temp = sklearn_K_means.fit_predict(image_sk).astype(np.int)
        if MASK:
            C = np.zeros(image.shape[:-1])[h:-h, w:-w] - 1
            C[mask] = temp
        else:
            C = temp
        C += 1
        C = C.reshape((n_r-2*h, n_c-2*w))

        # Save segmentations
        save_segmentation(FOLDER_RESULTS, '0_K_means_sklearn', C)

        # Save plot segmentations
        plot_segmentation(C, aspect=dataset.resolution[0]/dataset.resolution[1])
        save_figure(FOLDER_FIGURES, 'fig_K_means_sklearn')

        for i, features in enumerate(FEATURES_LIST):
            print('Features:', str(features))
            print()
            C = K_means_datacube(
                image,
                mask,
                features,
                WINDOWS_SHAPE,
                number_classes,
                NUMBER_INIT,
                K_MEANS_NB_ITER_MAX,
                EPS,
                ENABLE_MULTI,
                NUMBER_OF_THREADS_ROWS,
                NUMBER_OF_THREADS_COLUMNS
            )
            C = C.squeeze()
            C = C.astype(np.int)
         
            # Save segmentations
            save_segmentation(FOLDER_RESULTS, str(i+1) + '_K_means_' + str(features), C)

            # Save plot segmentations
            plot_segmentation(C, aspect=dataset.resolution[0]/dataset.resolution[1])
            save_figure(FOLDER_FIGURES, 'fig_K_means_' + str(features))

        t_end = time.time()
        print('TOTAL TIME ELAPSED:', round(t_end-t_beginning, 1), 's')
