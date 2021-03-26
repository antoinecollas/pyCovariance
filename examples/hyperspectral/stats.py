import autograd.numpy as np
import matplotlib.pyplot as plt
import os
import tikzplotlib

from pyCovariance.evaluation import create_directory

from pyCovariance.features.covariance import compute_scm

from pyCovariance.datasets.hyperspectral import Dataset


def main(
    dataset,
    border_size,
    mask,
    export_tex=False,
    verbose=True
):

    # folder path to save files
    folder = create_directory(dataset.name)

    image, gt = dataset.load(
        crop_image=False,
        border_size=border_size,
        pca=False,
        nb_bands_to_select=None
    )

    if mask:
        mask = gt >= 0
        image = image[mask]
    else:
        h, w, p = image.shape
        image = image.reshape((h*w, p))
    image = image.T

    if verbose:
        classes = np.unique(gt[gt >= 0])
        for i in classes:
            print('class', i, ':', np.sum(gt == i))
        print('total:', np.sum(gt >= 0))

    SCM = compute_scm(image)
    eigv = np.linalg.eigvalsh(SCM)[::-1]
    cumsum = np.cumsum(eigv) / np.sum(eigv)
    cumsum = np.concatenate([[0], cumsum])
    x = list(range(cumsum.shape[0]))

    plt.plot(x[:30], cumsum[:30], marker='+')
    path = os.path.join(folder, 'plot_cumulative_eigenvalues')
    if export_tex:
        tikzplotlib.save(path)
    plt.savefig(path)
    plt.close('all')

    plt.plot(x[1:31], eigv[:30], marker='+')
    path = os.path.join(folder, 'plot_eigenvalues')
    if export_tex:
        tikzplotlib.save(path)
    plt.savefig(path)
    plt.close('all')


if __name__ == '__main__':
    dataset_name = 'Indian_Pines'  # or 'Pavia' or 'Salinas'
    # border_size: discard 4 pixels around the image
    # used to compare with
    # different windows 5x5 vs 7x7 vs 9x9
    # 9//2 == 4
    border_size = 4
    dataset = Dataset(dataset_name)
    main(
        dataset=dataset,
        border_size=border_size,
        mask=True,
        export_tex=True
    )
