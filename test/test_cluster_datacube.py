import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd.numpy import random
import pytest
import os, sys, time

current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(current_dir)
sys.path.insert(1, temp)

from clustering_SAR.cluster_datacube import K_means_datacube
from clustering_SAR.evaluation import assign_classes_segmentation_to_gt, compute_mIoU
from clustering_SAR.features import Covariance, CovarianceEuclidean, CovarianceTexture, PixelEuclidean
from clustering_SAR.generation_data import generate_Toeplitz, sample_complex_normal

from clustering_SAR.generic_functions import plot_Pauli_SAR
from clustering_SAR.evaluation import plot_segmentation


def test_K_means_datacube():
    " Test function K_means_datacube on synthetic data."

    p = 3
    h, w = 30, 30

    # generation of two Toeplitz matrices
    rho = 0.1
    sigma_0 = generate_Toeplitz(rho, p)
    rho = 0.9
    sigma_1 = generate_Toeplitz(rho, p)

    # generation of the ground truth
    
    gt = np.ones((h, w))
    gt[int(h/5):int(h/2), int(w/5):int(w/2)] = 2
    
    # generation of the image
    image_0 = sample_complex_normal(h*w, sigma_0).T.reshape((h, w, p))
    image_1 = sample_complex_normal(h*w, sigma_1).T.reshape((h, w, p))
    image = image_0
    image[gt==1] = image_1[gt==1]

    # clustering
    WINDOWS_SHAPE = (3, 3)
    FEATURES_LIST = [Covariance(), CovarianceTexture(p=p, N=WINDOWS_SHAPE[0]*WINDOWS_SHAPE[1])]
    NUMBER_CLASSES = 2
    NUMBER_INIT = 1
    K_MEANS_NB_ITER_MAX = 100
    EPS = 1e-3
    ENABLE_MULTI = True
    NUMBER_OF_THREADS_ROWS = os.cpu_count()//2
    NUMBER_OF_THREADS_COLUMNS = 2
    
    h = WINDOWS_SHAPE[0]//2
    w = WINDOWS_SHAPE[1]//2
    gt = gt[h:-h, w:-w]
    
    for i, features in enumerate(FEATURES_LIST):
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        C = K_means_datacube(
            image,
            None,
            features,
            WINDOWS_SHAPE,
            NUMBER_CLASSES,
            NUMBER_INIT,
            K_MEANS_NB_ITER_MAX,
            EPS,
            ENABLE_MULTI,
            NUMBER_OF_THREADS_ROWS,
            NUMBER_OF_THREADS_COLUMNS,
        ).squeeze()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        C = assign_classes_segmentation_to_gt(C, gt)
        _, mIoU = compute_mIoU(C, gt)
        assert mIoU >= 0.8
