import autograd.numpy as np
import os

from pyCovariance.clustering_datacube import sliding_window_parallel
from pyCovariance.datasets.hyperspectral import Dataset
from pyCovariance.features import tau_UUH
from pyCovariance.utils import _compute_pairwise_distances


def main(
    dataset,
    crop_image,
    border_size,
    window_size,
    k,
    estimate_sigma,
    mask,
    n_jobs,
    verbose=True
):
    # load image and gt
    image, gt = dataset.load(
        crop_image=crop_image,
        pca=False
    )

    # mask
    mask_bool = mask
    mask = np.ones(gt.shape, dtype=bool)
    bs = border_size
    mask[:bs] = False
    mask[-bs:] = False
    mask[:, :bs] = False
    mask[:, -bs:] = False
    if mask_bool:
        mask[gt < 0] = False
    h = w = window_size//2
    mask = mask[h:-h, w:-w]

    if verbose:
        print('Crop image:', crop_image)
        print('image.shape:', image.shape)

    feature = tau_UUH(k, weights=(1, 1), estimate_sigma=estimate_sigma)
    feature = feature(p=dataset.dimension, N=window_size**2)

    X_temp = sliding_window_parallel(
        image=image,
        window_size=window_size,
        function_to_compute=feature.estimation,
        n_jobs_rows=n_jobs,
        n_jobs_columns=1,
        overlapping_window=True,
        verbose=False
    )
    X = None
    for row in range(len(X_temp)):
        for col in range(len(X_temp[row])):
            if X is None:
                X = X_temp[row][col]
            else:
                X.append(X_temp[row][col])
    mask = mask.reshape((-1)).astype(bool)
    X = X[mask]
    if verbose:
        print('X.shape:', X.shape)

    # compute mean distance between textures
    feature = tau_UUH(k, weights=(1, 0), estimate_sigma=estimate_sigma)
    feature = feature(p=dataset.dimension, N=window_size**2)
    d = _compute_pairwise_distances(
        X,
        X,
        distance_fct=feature.distance,
        n_jobs=n_jobs
    )
    d2_mean_tau = np.mean(d**2)
    if verbose:
        print('mean squared distances on texture:', d2_mean_tau)

    # compute mean distance between subspaces
    feature = tau_UUH(k, weights=(0, 1), estimate_sigma=estimate_sigma)
    feature = feature(p=dataset.dimension, N=window_size**2)
    d = _compute_pairwise_distances(
        X,
        X,
        distance_fct=feature.distance,
        n_jobs=n_jobs
    )
    d2_mean_U = np.mean(d**2)
    if verbose:
        print('mean squared distances on subspace:', d2_mean_U)


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    print('seed:', seed)

    dataset_name = 'Indian_Pines'
    # border_size: discard 4 pixels around the image
    # used to compare with
    # different windows 5x5 vs 7x7 vs 9x9
    # 9//2 == 4
    border_size = 4
    window_size = 7
    k = 5
    estimate_sigma = True
    dataset = Dataset(dataset_name)

    main(
        dataset=dataset,
        crop_image=True,
        border_size=border_size,
        window_size=window_size,
        k=k,
        estimate_sigma=estimate_sigma,
        mask=True,
        n_jobs=os.cpu_count()
    )
