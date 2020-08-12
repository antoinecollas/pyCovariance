import copy
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import sys

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(1, temp)

from clustering_SAR.generic_functions import assign_classes_segmentation_to_gt, compute_mIoU, plot_segmentation, plot_TP_FP_FN_segmentation

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
segmentation += 1

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

print()
print('################################################')
print('Supervised metric')
print('################################################')
old_segmentation = copy.deepcopy(segmentation)
IoU, mIoU = compute_mIoU(old_segmentation, gt, np.unique(segmentation))
print('mIoU before Hungarian algo=', mIoU)
segmentation = assign_classes_segmentation_to_gt(segmentation, gt, normalize=True)
IoU, mIoU = compute_mIoU(segmentation, gt, np.unique(segmentation))
print('mIoU=', mIoU)
classes = np.unique(segmentation).astype(np.int)
for i, class_ in enumerate(classes):
    print('class', class_, ': IoU=', round(IoU[i], 2))


print()
print('################################################')
print('Unsupervised metric')
print('################################################')
true = gt[gt!=0]
pred = segmentation[gt!=0]
AMI = adjusted_mutual_info_score(true, pred)
ARI = adjusted_rand_score(true, pred)
print('AMI=', AMI)
print('ARI=', ARI)

print()
print('################################################')
print('Plot')
print('################################################')
plot_segmentation(segmentation, classes=np.unique(gt).astype(np.int), title='Segmentation')
plot_segmentation(gt, title='Ground truth')
plot_TP_FP_FN_segmentation(segmentation, gt)
plt.show()
