from datetime import datetime
import glob
import numpy as np
import os
import random
from scipy.io import loadmat
import sys
import time

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(1, temp)

from clustering_SAR.cluster_datacube import K_means_datacube
from clustering_SAR.features import center_vectors_estimation, Covariance, CovarianceEuclidean, CovarianceTexture
from clustering_SAR.generic_functions import assign_classes_segmentation_to_gt, enable_latex_infigures, pca_and_save_variance, plot_segmentation, save_figure, save_segmentation

#######################################################
#######################################################
# BEGINNING OF HYPERPARAMETERS
#######################################################
#######################################################

# segmentation path (by default: get last folder)
FOLDER_REGEX = 'results/*'
folder_result = glob.glob(FOLDER_REGEX)[0]
segmentation = glob.glob(folder_result+'/*.npy')[0]
print('Segmentation file used :', segmentation)
segmentation = np.load(segmentation)
print(segmentation.shape)

# ground truth path
PATH_GT = 'data/Pavia/PaviaU_gt.mat'
gt = loadmat(PATH_GT)['paviaU_gt']
# Window size used to compute features
WINDOWS_SHAPE = (7,7)
h = WINDOWS_SHAPE[0]//2
w = WINDOWS_SHAPE[1]//2
gt = gt[h:-h, w:-w]

assert segmentation.shape == gt.shape

#######################################################
#######################################################
# END OF HYPERPARAMETERS
#######################################################
#######################################################

print('################################################')
print('Computing performances')
print('################################################')

new_segmentation = assign_classes_segmentation_to_gt(segmentation, gt)

import matplotlib.pyplot as plt
plot_segmentation(segmentation, classes=np.unique(gt))
plot_segmentation(new_segmentation, classes=np.unique(gt))
plot_segmentation(gt)
plt.show()
