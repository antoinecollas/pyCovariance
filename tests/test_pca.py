import autograd.numpy as np
from autograd.numpy import linalg as la, random as rnd
from sklearn.decomposition import PCA

from pyCovariance import pca_image
from pyCovariance.testing import assert_allclose


def test_real_pca():
    rnd.seed(123)

    H = 50
    W = 100
    p = 10
    nb_components = 3
    image = rnd.randn(H, W, p) + 4*np.ones((H, W, p))

    image_pca = pca_image(image, nb_components)
    assert image_pca.shape == (H, W, nb_components)
    assert image_pca.dtype == np.float64

    X = image.reshape((-1, p))
    pca = PCA()
    image_pca_sk = pca.fit_transform(X)
    image_pca_sk = image_pca_sk.reshape((H, W, p))
    image_pca_sk = image_pca_sk[:, :, :nb_components]

    # eigvenvectors have not necessarily same signs
    p1 = image_pca[0, 0]
    p2 = image_pca_sk[0, 0]
    sign = np.sign(p1/p2)
    image_pca_sk = image_pca_sk * sign

    assert_allclose(image_pca, image_pca_sk)


def test_complex_pca():
    rnd.seed(123)

    H = 50
    W = 100
    p = 10
    image = rnd.randn(H, W, p) + 1j*rnd.randn(H, W, p)
    image = image + 4*(np.ones((H, W, p)) + 1j*np.ones((H, W, p)))

    image_pca = pca_image(image, p)
    assert image_pca.shape == (H, W, p)
    assert image_pca.dtype == np.complex128

    X = image.reshape((-1, p)).conj().T
    mean = np.mean(X, axis=1).reshape((p, 1))
    X = X - mean
    SCM = (1/X.shape[1])*X@X.conj().T
    d, Q = la.eigh(SCM)
    reverse_idx = np.arange(len(d)-1, -1, step=-1)
    Q = Q[:, reverse_idx]
    image_pca = image_pca@Q.conj().T + mean.conj().T

    assert_allclose(image_pca, image)
