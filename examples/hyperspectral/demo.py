import autograd.numpy as np
import matplotlib.pyplot as plt
import os

from pyCovariance.datasets.hyperspectral import Dataset
from pyCovariance import pca_image
from pyCovariance.clustering_datacube import K_means_datacube
from pyCovariance.evaluation import\
        assign_segmentation_classes_to_gt_classes,\
        compute_OA,\
        plot_segmentation
from pyCovariance.features import covariance


def main(
    dataset_name='Indian_Pines',
    window_size=7,
    nb_bands=5,
    plot=True,
    crop_image=False,
    n_init=5,
    n_iter_max=100,
    verbose=True
):
    dataset = Dataset(dataset_name)

    # load image and gt
    image, gt = dataset.load(crop_image)
    n_r, n_c, p = image.shape
    nb_classes = np.sum(np.unique(gt) >= 0)

    # pca
    image = pca_image(image, nb_bands)

    # Riemannian clustering
    feature = covariance(nb_bands)
    C, _ = K_means_datacube(
        image,
        mask=None,
        features=feature,
        window_size=window_size,
        n_classes=nb_classes,
        n_init=n_init,
        n_iter_max=n_iter_max,
        eps=1e-2,
        nb_threads_rows=os.cpu_count()//2,
        nb_threads_columns=2,
        verbose=verbose
    )

    # evaluation
    h = w = window_size // 2
    gt = gt[h:-h, w:-w]
    C = assign_segmentation_classes_to_gt_classes(C, gt)
    OA = compute_OA(C, gt)
    if verbose:
        print('OA=', round(OA, 2))

    # plot
    plot_segmentation(C + 1, min_C=0, max_C=16)
    if plot:
        plt.show(block=False)
    else:
        plt.clf()
    plot_segmentation(gt + 1)
    if plot:
        plt.show()
    else:
        plt.clf()


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    print('seed:', seed)

    main()
