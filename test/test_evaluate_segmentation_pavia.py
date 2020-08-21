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

from clustering_SAR.evaluation import assign_classes_segmentation_to_gt, compute_mIoU, plot_segmentation

def test_assign_classes_segmentation_to_gt_and_compute_mIoU():
    # ground truth path
    PATH_GT = 'data/Pavia/PaviaU_gt.mat'
    gt = loadmat(PATH_GT)['paviaU_gt']

    _, mIoU = compute_mIoU(gt, gt, list(range(1, 10)))
    assert mIoU == 1

    # permutation of classes
    new_gt = gt
    while (new_gt == gt).any():
        classes = np.unique(gt)
        temp = np.random.permutation(classes)
        new_gt = np.zeros(gt.shape)
        for i, j in zip(classes, temp):
            new_gt[gt==i] = j
    
    _, mIoU = compute_mIoU(new_gt, gt, list(range(1, 10)))
    assert mIoU == 0
    
    back_to_gt = assign_classes_segmentation_to_gt(new_gt, gt)
    _, mIoU = compute_mIoU(back_to_gt, gt, list(range(1, 10)))
    
    assert mIoU == 1
    assert (back_to_gt == gt).all()
