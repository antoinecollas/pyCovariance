from datetime import datetime
import numpy as np
import os
import random
from scipy.io import loadmat
from sklearn.cluster import KMeans as sklearn_K_means
import sys
import time

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(1, temp)

from clustering_SAR.cluster_datacube import K_means_datacube
from clustering_SAR.features import center_vectors_estimation, Covariance, CovarianceEuclidean, CovarianceTexture, PixelEuclidean
from clustering_SAR.generic_functions import enable_latex_infigures, pca_and_save_variance, save_figure
from clustering_SAR.evaluation import plot_segmentation, save_segmentation

#######################################################
#######################################################
# BEGINNING OF HYPERPARAMETERS
#######################################################
#######################################################

DEBUG = False
if DEBUG:
    print()
    print('DEBUG mode enabled !!!')
    SIZE_CROP = 100

# folder to save results
date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
FOLDER_RESULTS = os.path.join('results', 'Pavia_'+date_str)
FOLDER_FIGURES = os.path.join(FOLDER_RESULTS, 'figures')

# Activate latex in figures (or not)
LATEX_IN_FIGURES = False
if LATEX_IN_FIGURES:
  enable_latex_infigures()

# Enable parallel processing (or not)
ENABLE_MULTI = True
NUMBER_OF_THREADS_ROWS = os.cpu_count()//2
NUMBER_OF_THREADS_COLUMNS = 2
if NUMBER_OF_THREADS_ROWS*NUMBER_OF_THREADS_COLUMNS != os.cpu_count():
    print('ERROR: all cpus are not used ...')
    sys.exit(1)
NUMBER_OF_THREADS = os.cpu_count() 

# Dataset
PATH = 'data/Pavia/PaviaU.mat'
KEY_DICT_PAVIA = 'paviaU'
NUMBER_CLASSES = 9
NB_BANDS_TO_SELECT = 4
RESOLUTION = [1.3, 1.3] # resolution in meters

# Window size to compute features
WINDOWS_SHAPE = (3,3)

# features used to cluster the image
#features_list = [PixelEuclidean(), CovarianceEuclidean(), Covariance(), CovarianceTexture(p=NB_BANDS_TO_SELECT, N=WINDOWS_SHAPE[0]*WINDOWS_SHAPE[1])]
features_list = [PixelEuclidean()]

# K-means parameter
if DEBUG:
    NUMBER_INIT = 1
    K_MEANS_NB_ITER_MAX = 2
else:
    NUMBER_INIT = 10
    K_MEANS_NB_ITER_MAX = 100
EPS = 1e-3

#######################################################
#######################################################
# END OF HYPERPARAMETERS
#######################################################
#######################################################

print('################################################')
print('Reading dataset') 
print('################################################')
t_beginning = time.time()

# load image
image = loadmat(PATH)[KEY_DICT_PAVIA]

# center image globally
mean = np.mean(image, axis=0)
image = image - mean
# check pixels are centered
assert (np.abs(np.mean(image, axis=0)) < 1e-9).all()

# pca
image = pca_and_save_variance(FOLDER_FIGURES, 'fig_explained_variance_Pavia', image, NB_BANDS_TO_SELECT)

if DEBUG:
    center = np.array(image.shape[0:2])//2
    half_height = SIZE_CROP//2
    half_width = SIZE_CROP//2
    image = image[center[0]-half_height:center[0]+half_height, center[1]-half_width:center[1]+half_width]
n_r, n_c, p = image.shape
print('image.shape', image.shape)

print()
print('K-means using Sklearn implementation ...') 
print()

# We use scikit-learn K-means implementation as a reference
n_jobs = NUMBER_OF_THREADS_ROWS*NUMBER_OF_THREADS_COLUMNS if ENABLE_MULTI else 1
sklearn_K_means = sklearn_K_means(n_clusters=NUMBER_CLASSES, n_init=NUMBER_INIT)
C = sklearn_K_means.fit_predict(image.reshape((-1, NB_BANDS_TO_SELECT)))
C = C.reshape((n_r, n_c))
h = WINDOWS_SHAPE[0]//2
w = WINDOWS_SHAPE[1]//2
C = C[h:-h, w:-w]
C = C.astype(np.int)
C = C + 1

# Save segmentations
save_segmentation(FOLDER_RESULTS, '0_K_means_sklearn_Pavia', C)

# Save plot segmentations
plot_segmentation(C, aspect=RESOLUTION[0]/RESOLUTION[1])
save_figure(FOLDER_FIGURES, 'fig_K_means_sklearn_Pavia')

for i, features in enumerate(features_list):
    print('Features:', str(features))
    print()
    C = K_means_datacube(
        image,
        features,
        WINDOWS_SHAPE,
        NUMBER_CLASSES,
        NUMBER_INIT,
        K_MEANS_NB_ITER_MAX,
        EPS,
        ENABLE_MULTI,
        NUMBER_OF_THREADS_ROWS,
        NUMBER_OF_THREADS_COLUMNS
    )
    C = C.squeeze()
    C = C.astype(np.int)
    C = C + 1
 
    # Save segmentations
    save_segmentation(FOLDER_RESULTS, str(i+1) + '_K_means_' + str(features) + '_Pavia', C)

    # Save plot segmentations
    plot_segmentation(C, aspect=RESOLUTION[0]/RESOLUTION[1])
    save_figure(FOLDER_FIGURES, 'fig_K_means_' + str(features) + '_Pavia')

t_end = time.time()
print('TOTAL TIME ELAPSED:', round(t_end-t_beginning, 1), 's')
