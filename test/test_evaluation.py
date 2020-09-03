import glob
import numpy as np
from numpy import random
import os
from scipy.io import loadmat
from sklearn.metrics import f1_score
import sys

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(1, temp)

from clustering_SAR.evaluation import assign_classes_segmentation_to_gt, compute_mIoU

def test_assign_classes_segmentation_to_gt_and_compute_mIoU():
    # generation of ground truth
    gt = np.zeros(1000)
    for i in range(len(gt)):
        a = random.random_sample()
        if a < 0.1:
            gt[i] = 1
        elif a < 0.5:
            gt[i] = 2
        else:
            gt[i] = 3

    # permutation of classes
    C = gt
    while (C == gt).any():
        classes = np.unique(gt)
        temp = np.random.permutation(classes)
        C = np.zeros(gt.shape)
        for i, j in zip(classes, temp):
            C[gt == i] = j
    
    for i in range(len(C)):
        a = random.random_sample()
        if a < 0.05:
            C[i] = C[i] + 1
        if C[i] == 4:
            C[i] = 1

    C = assign_classes_segmentation_to_gt(C, gt)
    assert f1_score(gt, C, average='macro') > 0.8
    _, mIoU = compute_mIoU(C, gt)
    assert mIoU > 0.8
