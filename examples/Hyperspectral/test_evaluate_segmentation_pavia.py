import glob
import numpy as np
import os
from scipy.io import loadmat
import sys

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(1, temp)

from clustering_SAR.generic_functions import assign_classes_segmentation_to_gt, plot_segmentation

#######################################################
#######################################################
# BEGINNING OF HYPERPARAMETERS
#######################################################
#######################################################

# ground truth path
PATH_GT = 'data/Pavia/PaviaU_gt.mat'
gt = loadmat(PATH_GT)['paviaU_gt']
# Window size used to compute features
WINDOWS_SHAPE = (7,7)
h = WINDOWS_SHAPE[0]//2
w = WINDOWS_SHAPE[1]//2
gt = gt[h:-h, w:-w]

#######################################################
#######################################################
# END OF HYPERPARAMETERS
#######################################################
#######################################################

print('################################################')
print('Test of the Hungarian algorithm')
print('################################################')

classes = np.unique(gt)
temp = np.random.permutation(classes)
print('original classes:', classes)
print('new classes:', temp)

new_gt = np.zeros(gt.shape)
for i, j in zip(classes, temp):
    new_gt[gt==i] = j
back_to_gt = assign_classes_segmentation_to_gt(new_gt, gt)

import matplotlib.pyplot as plt
plot_segmentation(gt)
plot_segmentation(new_gt)
plot_segmentation(back_to_gt)
plt.show()
