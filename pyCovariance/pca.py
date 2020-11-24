import autograd.numpy as np
from autograd.numpy import linalg as la


def pca_image(image, nb_components):
    """ A function that centers data and applies PCA on an image.
        Inputs:
            * image: numpy array of the image.
            * nb_components: number of components to keep.
    """
    # center pixels
    h, w, p = image.shape
    X = image.reshape((h*w, p))
    mean = np.mean(X, axis=0)
    image = image - mean
    X = X - mean
    # check pixels are centered
    assert (np.abs(np.mean(X, axis=0)) < 1e-10).all()

    # apply PCA
    SCM = (1/len(X))*X.conj().T@X
    d, Q = la.eigh(SCM)
    reverse_idx = np.arange(len(d)-1, -1, step=-1)
    Q = Q[:, reverse_idx]
    Q = Q[:, :nb_components]
    image = image@Q

    return image
