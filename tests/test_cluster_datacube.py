import autograd.numpy as np
import os

from pyCovariance import K_means_datacube
from pyCovariance.features.base import _FeatureArray
from pyCovariance.clustering_datacube import sliding_window_parallel
from pyCovariance.features import center_euclidean
from pyCovariance.generation_data import\
        generate_complex_covariance,\
        generate_covariance,\
        sample_complex_normal_distribution,\
        sample_normal_distribution

# from pyCovariance.evaluation import plot_segmentation


def test_sliding_window_parallel():
    p = 3
    H = 50
    W = 100

    # generation of the image
    cov = generate_complex_covariance(p)
    temp = sample_complex_normal_distribution(H*W, cov)
    image = temp.reshape((H, W, p))
    assert image.shape == (H, W, p)
    assert image.dtype == np.complex128

    window_size = 3
    fct = center_euclidean(p).estimation
    res = sliding_window_parallel(
        image,
        window_size,
        fct,
        nb_threads_rows=os.cpu_count()//2,
        nb_threads_columns=2,
        overlapping_window=True,
        verbose=False
    )
    assert type(res) == list
    assert type(res[0]) == list
    assert type(res[0][0]) == _FeatureArray

    h = w = window_size//2
    image = image[h:-h, w:-w]
    assert (len(res), len(res[0])) == image.shape[:2]

    for i in range(len(res)):
        for j in range(len(res[0])):
            assert (image[i, j] == res[i][j].export()).all()


def test_real_K_means_datacube():
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
    WINDOW_SIZE = 3
    MASK = None
    FEATURE = center_euclidean(p)
    NUMBER_CLASSES = 2
    NUMBER_INIT = 1
    K_MEANS_NB_ITER_MAX = 100
    EPS = 1e-3
    NUMBER_OF_THREADS_ROWS = 1
    NUMBER_OF_THREADS_COLUMNS = 1

    h = w = WINDOW_SIZE//2
    gt = gt[h:-h, w:-w]

    C, criterion_values = K_means_datacube(
        image,
        MASK,
        FEATURE,
        WINDOW_SIZE,
        NUMBER_CLASSES,
        NUMBER_INIT,
        K_MEANS_NB_ITER_MAX,
        EPS,
        NUMBER_OF_THREADS_ROWS,
        NUMBER_OF_THREADS_COLUMNS,
        verbose=False
    )

    assert C.shape == gt.shape
    assert C.dtype == np.int64
    assert type(criterion_values) == list

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

    C, criterion_values = K_means_datacube(
        image,
        MASK,
        FEATURE,
        WINDOW_SIZE,
        NUMBER_CLASSES,
        NUMBER_INIT,
        K_MEANS_NB_ITER_MAX,
        EPS,
        NUMBER_OF_THREADS_ROWS,
        NUMBER_OF_THREADS_COLUMNS,
        verbose=False
    )

    assert C.shape == gt.shape
    assert C.dtype == np.int64
    assert type(criterion_values) == list

    MASK = MASK[h:-h, w:-w].astype(bool)
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
    WINDOW_SIZE = 3
    MASK = None
    FEATURE = center_euclidean(p)
    NUMBER_CLASSES = 2
    NUMBER_INIT = 1
    K_MEANS_NB_ITER_MAX = 100
    EPS = 1e-3
    NUMBER_OF_THREADS_ROWS = os.cpu_count()//2
    NUMBER_OF_THREADS_COLUMNS = 2

    C, criterion_values = K_means_datacube(
        image,
        MASK,
        FEATURE,
        WINDOW_SIZE,
        NUMBER_CLASSES,
        NUMBER_INIT,
        K_MEANS_NB_ITER_MAX,
        EPS,
        NUMBER_OF_THREADS_ROWS,
        NUMBER_OF_THREADS_COLUMNS,
        verbose=False
    )

    assert C.shape == gt.shape
    assert C.dtype == np.int64
    assert type(criterion_values) == list

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


def test_complex_K_means_datacube():
    # test K_means_datacube on complex valued synthetic data

    p = 3
    H = 50
    W = 100

    # generation of the ground truth
    gt = np.zeros((H, W))
    gt[:, int(W/2):] = 1

    # generation of the image
    cov1 = generate_complex_covariance(p)
    temp1 = sample_complex_normal_distribution(H*int(W/2), cov1)
    temp1 = temp1 + (np.ones((p, 1))+1j*np.ones((p, 1)))
    temp1 = temp1.conj().T
    cov2 = generate_complex_covariance(p)
    temp2 = sample_complex_normal_distribution(H*int(W/2), cov2)
    temp2 = temp2 - (np.ones((p, 1))+1j*np.ones((p, 1)))
    temp2 = temp2.conj().T
    image = np.zeros((H, W, p), dtype=np.complex128)
    image[:, :int(W/2)] = temp1.reshape((H, int(W/2), p))
    image[:, int(W/2):] = temp2.reshape((H, int(W/2), p))

    # clustering with one thread
    WINDOW_SIZE = 3
    MASK = None
    FEATURE = center_euclidean(p)
    NUMBER_CLASSES = 2
    NUMBER_INIT = 1
    K_MEANS_NB_ITER_MAX = 100
    EPS = 1e-3
    NUMBER_OF_THREADS_ROWS = 1
    NUMBER_OF_THREADS_COLUMNS = 1

    h = w = WINDOW_SIZE//2
    gt = gt[h:-h, w:-w]

    C, criterion_values = K_means_datacube(
        image,
        MASK,
        FEATURE,
        WINDOW_SIZE,
        NUMBER_CLASSES,
        NUMBER_INIT,
        K_MEANS_NB_ITER_MAX,
        EPS,
        NUMBER_OF_THREADS_ROWS,
        NUMBER_OF_THREADS_COLUMNS,
        verbose=False
    )

    assert C.shape == gt.shape
    assert C.dtype == np.int64
    assert type(criterion_values) == list

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
