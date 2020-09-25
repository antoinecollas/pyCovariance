import autograd.numpy as np
from sklearn.decomposition import PCA


def pca(image, nb_components):
    """ A function that centers data and applies PCA.
        Inputs:
            * image: numpy array of the image.
            * nb_components: number of components to keep.
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

    return image
