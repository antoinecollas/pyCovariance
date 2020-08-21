import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
import tikzplotlib


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


def enable_latex_infigures():
    """ A function that allows to enable latex in figures"""
    from matplotlib import rc
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"


def vec(mat):
    return mat.ravel('F')


def vech(mat):
    # Gets Fortran-order
    rows, cols = np.triu_indices(len(mat))
    vec = mat[rows, cols]
    return vec


def _diag_indices(n):
    rows, cols = np.diag_indices(n)
    return rows * n + cols


def unvec(v):
    k = int(np.sqrt(len(v)))
    assert(k * k == len(v))
    return v.reshape((k, k), order='F')


def unvech(v):
    # quadratic formula, correct fp error
    rows = .5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = np.zeros((rows, rows), dtype=v.dtype)
    result[np.triu_indices(rows)] = v
    result = result + result.conj().T

    # divide diagonal elements by 2
    result[np.diag_indices(rows)] /= 2

    return result


def plot_Pauli_SAR(image, aspect=1):
    """ 1st dimension =HH, 2nd dimnension = HV, 3rd dimension=VV"""
    R = np.abs(image[:,:,0] - image[:,:,2])
    G = np.abs(image[:,:,1])
    B = np.abs(image[:,:,0] + image[:,:,2])
    fig = plt.figure()
    RGB_image = np.stack([R,G,B], axis=2)
    RGB_image = RGB_image - np.min(RGB_image)
    RGB_image[RGB_image > 1] = 1
    plt.imshow(RGB_image, aspect=aspect)
    plt.axis('off')
    return fig


def save_figure(folder, figname):
    """ A function that save the current figure in '.png' and in '.tex'.
        Inputs:
            * folder: string corresponding to the folder's name where to save the actual figure.
            * figname: string corresponding to the name of the figure to save.
    """
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, figname)
    
    path_png = path + '.png'
    plt.savefig(path_png)

    path_tex = path + '.tex'
    tikzplotlib.save(path_tex)


def pca_and_save_variance(folder, figname, image, nb_components):
    """ A function that centers data and applies PCA. It also saves a figure of the explained variance.
        Inputs:
            * folder: string.
            * figname: string.
            * image: numpy array to save.
    """
    # center pixels
    h, w, p = image.shape
    image = image.reshape((-1, image.shape[-1]))
    mean = np.mean(image, axis=0)
    image = image - mean
    # check pixels are centered
    assert (np.abs(np.mean(image, axis=0)) < 1e-9).all()

    # apply PCA
    pca = PCA()
    image = pca.fit_transform(image)
    # check pixels are still centered
    assert (np.abs(np.mean(image, axis=0)) < 1e-9).all()
    # reshape image
    image = image.reshape((h, w, p))
    image = image[:, :, :nb_components]

    # plot and save explained variance 
    plt.plot(np.arange(1, p+1), np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance');
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, figname)
    path_png = path + '.png'
    plt.savefig(path_png)
    path_tex = path + '.tex'
    tikzplotlib.save(path_tex)
    
    return image
