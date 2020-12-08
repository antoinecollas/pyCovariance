import autograd.numpy as np
from autograd.numpy import random as rnd
from copy import deepcopy
from sklearn.metrics import f1_score

from pyCovariance.evaluation import\
        assign_segmentation_classes_to_gt_classes,\
        compute_mIoU,\
        compute_OA


def test_assign_segmentation_classes_to_gt_classes_and_compute_mIoU():
    N = int(1e3)

    # generation of ground truth
    gt = np.zeros(N)
    for i in range(len(gt)):
        a = rnd.random_sample()
        if a < 0.1:
            gt[i] = 0
        elif a < 0.5:
            gt[i] = 1
        else:
            gt[i] = 2

    # permutation of classes
    C = gt
    classes = np.unique(gt)
    while (C == gt).any():
        temp = rnd.permutation(classes)
        C = deepcopy(gt)
        for i, j in zip(classes, temp):
            C[gt == i] = j

    assert f1_score(gt[gt >= 0], C[gt >= 0], average='macro') == 0
    _, mIoU = compute_mIoU(C, gt)
    assert mIoU == 0
    OA = compute_OA(C, gt)
    assert OA == 0

    C_temp = assign_segmentation_classes_to_gt_classes(C, gt, normalize=True)
    assert f1_score(gt[gt >= 0], C_temp[gt >= 0], average='macro') == 1
    _, mIoU = compute_mIoU(C_temp, gt)
    assert mIoU == 1
    OA = compute_OA(C_temp, gt)
    assert OA == 1

    # we add wrong classifications
    for i in range(len(C)):
        a = rnd.random_sample()
        if a < 0.05:
            C[i] = rnd.choice([0, 1, 2])

    assert f1_score(gt, C, average='macro') < 0.1
    _, mIoU = compute_mIoU(C, gt)
    assert mIoU < 0.1

    # we change the number of classes in C
    C[C == 0] = 8
    C[C == 1] = 9
    C[C == 2] = 27

    C = assign_segmentation_classes_to_gt_classes(C, gt)
    assert f1_score(gt, C, average='macro') > 0.85
    _, mIoU = compute_mIoU(C, gt)
    assert mIoU > 0.85
    OA = compute_OA(C, gt)
    assert OA > 0.85

    # test with negative classes (these classes should be ignored!)
    # generation of ground truth
    gt = np.zeros(N)
    for i in range(len(gt)):
        a = rnd.random_sample()
        if a < 0.8:
            gt[i] = -3
        elif a < 0.9:
            gt[i] = 0
        elif a < 0.95:
            gt[i] = 2
        else:
            gt[i] = 3

    # permutation of classes
    C = gt
    classes = np.unique(gt[gt >= 0])
    while (C == gt)[gt >= 0].any():
        temp = rnd.permutation(classes)
        C = deepcopy(gt)
        for i, j in zip(classes, temp):
            C[gt == int(i)] = int(j)

    assert f1_score(gt[gt >= 0], C[gt >= 0], average='macro') == 0
    _, mIoU = compute_mIoU(C, gt)
    assert mIoU == 0
    OA = compute_OA(C, gt)
    assert OA == 0

    C_temp = assign_segmentation_classes_to_gt_classes(C, gt)
    assert f1_score(gt[gt >= 0], C_temp[gt >= 0], average='macro') == 1
    _, mIoU = compute_mIoU(C_temp, gt)
    assert mIoU == 1
    OA = compute_OA(C_temp, gt)
    assert OA == 1

    # we add wrong classifications
    for i in range(len(C)):
        a = rnd.random_sample()
        if a < 0.05:
            C[i] = rnd.choice([0, 2, 3])

    assert f1_score(gt[gt >= 0], C[gt >= 0], average='macro') < 0.1
    _, mIoU = compute_mIoU(C, gt)
    assert mIoU < 0.1
    OA = compute_OA(C, gt)
    assert OA < 0.1

    C = assign_segmentation_classes_to_gt_classes(C, gt)
    assert f1_score(gt[gt >= 0], C[gt >= 0], average='macro') > 0.85
    _, mIoU = compute_mIoU(C, gt)
    assert mIoU > 0.85
    OA = compute_OA(C, gt)
    assert OA > 0.85
