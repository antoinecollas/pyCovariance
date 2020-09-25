import autograd.numpy as np
import matplotlib.pyplot as plt
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
