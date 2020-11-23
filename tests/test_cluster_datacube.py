import autograd.numpy as np
import os
import sys

from pyCovariance.cluster_datacube import K_means_datacube
from pyCovariance.features import pixel_euclidean
from pyCovariance.generation_data import\
        generate_covariance,\
        sample_normal_distribution

# from pyCovariance.evaluation import plot_segmentation


def test_K_means_datacube():
    # test K_means_datacube on synthetic data

    p = 3
    H = 50
    W = 100

    # generation of the ground truth
    gt = np.zeros((H, W))
    gt[:, int(W/2):] = 1

    # generation of the image
    cov1 = generate_covariance(p)
    temp1 = sample_normal_distribution(H*int(W/2), cov1) + 2*np.ones((p, 1))
    temp1 = temp1.T
    cov2 = generate_covariance(p)
    temp2 = sample_normal_distribution(H*int(W/2), cov2) - 2*np.ones((p, 1))
    temp2 = temp2.T
    image = np.zeros((H, W, p))
    image[:, :int(W/2)] = temp1.reshape((H, int(W/2), p))
    image[:, int(W/2):] = temp2.reshape((H, int(W/2), p))

    # clustering with one thread
    WINDOWS_SHAPE = (3, 3)
    MASK = None
    FEATURE = pixel_euclidean(p)
    NUMBER_CLASSES = 2
    NUMBER_INIT = 1
    K_MEANS_NB_ITER_MAX = 100
    EPS = 1e-3
    ENABLE_MULTI = False
    NUMBER_OF_THREADS_ROWS = 1
    NUMBER_OF_THREADS_COLUMNS = 1

    h = WINDOWS_SHAPE[0]//2
    w = WINDOWS_SHAPE[1]//2
    gt = gt[h:-h, w:-w]

    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    C = K_means_datacube(
        image,
        MASK,
        FEATURE,
        WINDOWS_SHAPE,
        NUMBER_CLASSES,
        NUMBER_INIT,
        K_MEANS_NB_ITER_MAX,
        EPS,
        ENABLE_MULTI,
        NUMBER_OF_THREADS_ROWS,
        NUMBER_OF_THREADS_COLUMNS,
    )
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    assert C.shape == gt.shape
    assert C.dtype == np.int64

    # import matplotlib.pyplot as plt
    # plot_segmentation(C)
    # plt.show()
    # plot_segmentation(gt)
    # plt.show()

    y = gt.reshape(-1)
    y_predict = C.reshape(-1)
    precision = np.sum(y == y_predict)/len(y)
    if precision < 0.5:
        y_predict = np.mod(y_predict+1, 2)
    precision = np.sum(y == y_predict)/len(y)
    assert precision >= 0.95

    # clustering within a mask
    MASK = np.zeros((H, W))
    MASK[int(H/2)-10:int(H/2)+10, int(W/2)-10:int(W/2)+10] = 1
    MASK = MASK[h:-h, w:-w]

    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    C = K_means_datacube(
        image,
        MASK,
        FEATURE,
        WINDOWS_SHAPE,
        NUMBER_CLASSES,
        NUMBER_INIT,
        K_MEANS_NB_ITER_MAX,
        EPS,
        ENABLE_MULTI,
        NUMBER_OF_THREADS_ROWS,
        NUMBER_OF_THREADS_COLUMNS,
    )
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    assert C.shape == gt.shape
    assert C.dtype == np.int64

    MASK = MASK.astype(bool)
    assert (C[~MASK] == -1).all()
    assert (C[MASK] != -1).all()

    y = gt[MASK].reshape(-1)
    y_predict = C[MASK].reshape(-1)
    precision = np.sum(y == y_predict)/len(y)
    if precision < 0.5:
        y_predict = np.mod(y_predict+1, 2)
    precision = np.sum(y == y_predict)/len(y)
    assert precision >= 0.95

    # clustering with multiple threads
    WINDOWS_SHAPE = (3, 3)
    MASK = None
    FEATURE = pixel_euclidean(p)
    NUMBER_CLASSES = 2
    NUMBER_INIT = 1
    K_MEANS_NB_ITER_MAX = 100
    EPS = 1e-3
    ENABLE_MULTI = True
    NUMBER_OF_THREADS_ROWS = os.cpu_count()//2
    NUMBER_OF_THREADS_COLUMNS = 2

    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    C = K_means_datacube(
        image,
        MASK,
        FEATURE,
        WINDOWS_SHAPE,
        NUMBER_CLASSES,
        NUMBER_INIT,
        K_MEANS_NB_ITER_MAX,
        EPS,
        ENABLE_MULTI,
        NUMBER_OF_THREADS_ROWS,
        NUMBER_OF_THREADS_COLUMNS,
    )
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    assert C.shape == gt.shape
    assert C.dtype == np.int64

    # import matplotlib.pyplot as plt
    # plot_segmentation(C)
    # plt.show()
    # plot_segmentation(gt)
    # plt.show()

    y = gt.reshape(-1)
    y_predict = C.reshape(-1)
    precision = np.sum(y == y_predict)/len(y)
    if precision < 0.5:
        y_predict = np.mod(y_predict+1, 2)
    precision = np.sum(y == y_predict)/len(y)
    assert precision >= 0.95
