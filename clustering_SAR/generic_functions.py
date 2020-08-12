import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import scipy.special
from sklearn.decomposition import PCA
import sys
import tikzplotlib
import warnings

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


def multivariate_complex_normal_samples(mean, covariance, N, pseudo_covariance=0):
    """ A function to generate multivariate complex normal vectos as described in:
        Picinbono, B. (1996). Second-order complex random vectors and normal
        distributions. IEEE Transactions on Signal Processing, 44(10), 2637–2640.
        Inputs:
            * mean = vector of size p, mean of the distribution
            * covariance = the covariance matrix of size p*p(Gamma in the paper)
            * pseudo_covariance = the pseudo-covariance of size p*p (C in the paper)
                for a circular distribution omit the parameter
            * N = number of Samples
        Outputs:
            * Z = Samples from the complex Normal multivariate distribution, size p*N"""

    (p, p) = covariance.shape
    Gamma = covariance
    C = pseudo_covariance

    # Computing elements of matrix Gamma_2r
    Gamma_x = 0.5 * np.real(Gamma + C)
    Gamma_xy = 0.5 * np.imag(-Gamma + C)
    Gamma_yx = 0.5 * np.imag(Gamma + C)
    Gamma_y = 0.5 * np.real(Gamma - C)

    # Matrix Gamma_2r as a block matrix
    Gamma_2r = np.block([[Gamma_x, Gamma_xy], [Gamma_yx, Gamma_y]])

    # Generating the real part and imaginary part
    mu = np.hstack((mean.real, mean.imag))
    v = np.random.multivariate_normal(mu, Gamma_2r, N).T
    X = v[0:p, :]
    Y = v[p:, :]
    return X + 1j * Y


def multivariate_complex_t_samples(mean, covariance, N, df, pseudo_covariance=0):
    """ A function to generate multivariate complex t distributed vectors using the
    definition with a product of a multivaraite normal with an inverse chi2 distributed samples. 
    Inputs:
        * mean = vector of size p, mean of the distribution
        * covariance = the covariance matrix of size p*p
        * pseudo_covariance = the pseudo-covariance of size p*p
            for a circular distribution omit the parameter
        * df = degrees of freedom of the chi-squared distribution
        * N = number of Samples
    Outputs:
        * Z = Samples from the complex multivariate t distribution, size p*N"""


    if df == np.inf:
        x = 1
    else:
        x = np.random.chisquare(df, N)/df
    z = multivariate_complex_normal_samples(np.zeros(mean.shape), covariance, N, pseudo_covariance)
    return np.tile(mean.reshape((len(mean),1)),(1,N)) + z/np.sqrt(x)[None,:] 


def multivariate_complex_K_samples(mean, covariance, N, mu, b, pseudo_covariance=0):
    """ A function to generate multivariate complex K distributed vectors using the
    definition provided at page 27 of the Pd.d thesis:
    "Detection en environement non Gaussien", Emanuelle Jay. 
    Inputs:
        * mean = vector of size p, mean of the distribution
        * covariance = the covariance matrix of size p*p
        * pseudo_covariance = the pseudo-covariance of size p*p
            for a circular distribution omit the parameter
        * mu = Shape parameter
        * b = Scale parameter
        * N = number of Samples
    Outputs:
        * Z = Samples from the complex multivariate t distribution, size p*N"""

    x = np.random.gamma(mu, 2/(b**2), N)
    z = multivariate_complex_normal_samples(np.zeros(mean.shape), covariance, N, pseudo_covariance)
    return np.tile(mean.reshape((len(mean),1)),(1,N)) + z*np.sqrt(x)[None,:]   


def multivariate_complex_Cauchy_samples(mean, covariance, N, mu, b, pseudo_covariance=0):
    """ A function to generate multivariate complex Cauchy distributed vectors using the
    definition provided at page 26 of the Pd.d thesis:
    "Detection en environement non Gaussien", Emanuelle Jay. 
    Inputs:
        * mean = vector of size p, mean of the distribution
        * covariance = the covariance matrix of size p*p
        * pseudo_covariance = the pseudo-covariance of size p*p
            for a circular distribution omit the parameter
        * mu = Shape parameter
        * b = Scale parameter
        * N = number of Samples
    Outputs:
        * Z = Samples from the complex multivariate t distribution, size p*N"""

    x = np.random.gamma(mu, 2/(b**2), N)
    z = multivariate_complex_normal_samples(np.zeros(mean.shape), covariance, N, pseudo_covariance)
    return np.tile(mean.reshape((len(mean),1)),(1,N)) + z/np.sqrt(x)[None,:]    


def multivariate_complex_Laplace_samples(mean, covariance, N, beta, pseudo_covariance=0):
    """ A function to generate multivariate complex Cauchy distributed vectors using the
    definition provided at page 27 of the Pd.d thesis:
    "Detection en environement non Gaussien", Emanuelle Jay. 
    Inputs:
        * mean = vector of size p, mean of the distribution
        * covariance = the covariance matrix of size p*p
        * pseudo_covariance = the pseudo-covariance of size p*p
            for a circular distribution omit the parameter
        * beta = Scale parameter
        * N = number of Samples
    Outputs:
        * Z = Samples from the complex multivariate t distribution, size p*N"""

    x = np.random.exponential(beta, N)
    z = multivariate_complex_normal_samples(np.zeros(mean.shape), covariance, N, pseudo_covariance)
    return np.tile(mean.reshape((len(mean),1)),(1,N)) + z*np.sqrt(x)[None,:]    


def SCM(x, *args):
    """ A function that computes the SCM for covariance matrix estimation
            Inputs:
                * x = a matrix of size p*N with each observation along column dimension
            Outputs:
                * Sigma = the estimate"""

    (p, N) = x.shape
    return (x @ x.conj().T) / N


def tyler_estimator_covariance(𝐗, tol=0.001, iter_max=20):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    𝚺 = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (δ>tol) and (iteration<iter_max):
        
        # Computing expression of Tyler estimator (with matrix multiplication)
        τ = np.diagonal(𝐗.conj().T@np.linalg.inv(𝚺)@𝐗)
        𝐗_bis = 𝐗 / np.sqrt(τ)
        𝚺_new = (p/N) * 𝐗_bis@𝐗_bis.conj().T

        # Imposing trace constraint: Tr(𝚺) = p
        𝚺_new = p*𝚺_new/np.trace(𝚺_new)

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')
        iteration = iteration + 1

        # Updating 𝚺
        𝚺 = 𝚺_new

    if iteration == iter_max:
        warnings.warn('Recursive algorithm did not converge')

    return (𝚺, δ, iteration)


def tyler_estimator_covariance_normalisedet(𝐗, tol=0.001, iter_max=20):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        and normalisation by determinant
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    𝚺 = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (δ>tol) and (iteration<iter_max):
        
        # Computing expression of Tyler estimator (with matrix multiplication)
        τ = np.diagonal(𝐗.conj().T@np.linalg.inv(𝚺)@𝐗)
        𝐗_bis = 𝐗 / np.sqrt(τ)
        𝚺_new = (p/N) * 𝐗_bis@𝐗_bis.conj().T

        # # Imposing det constraint: det(𝚺) = 1 DOT NOT WORK HERE
        # 𝚺 = 𝚺/(np.linalg.det(𝚺)**(1/p))

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')
        iteration = iteration + 1

        # Updating 𝚺
        𝚺 = 𝚺_new

    # Imposing det constraint: det(𝚺) = 1
    𝚺 = 𝚺/(np.linalg.det(𝚺)**(1/p))

    if iteration == iter_max:
        warnings.warn('Recursive algorithm did not converge')

    return (𝚺, δ, iteration)


def student_t_estimator_covariance_mle(𝐗, d, tol=0.001, iter_max=20):
    """ A function that computes the MLE for covariance matrix estimation for a student t distribution
        when the degree of freedom is known
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    𝚺 = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (δ>tol) and (iteration<iter_max):
        
        # Computing expression of Tyler estimator (with matrix multiplication)
        τ = d + np.diagonal(𝐗.conj().T@np.linalg.inv(𝚺)@𝐗)
        𝐗_bis = 𝐗 / np.sqrt(τ)
        𝚺_new = ((d+p)/N) * 𝐗_bis@𝐗_bis.conj().T

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')
        iteration = iteration + 1

        # Updating 𝚺
        𝚺 = 𝚺_new

    if iteration == iter_max:
        warnings.warn('Recursive algorithm did not converge')

    return (𝚺, δ, iteration)

def ToeplitzMatrix(rho, p):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * rho = a scalar
                * p = size of matrix
            Outputs:
                * the matrix """

    return sp.linalg.toeplitz(np.power(rho, np.arange(0, p)))

def vec(mat):
    return mat.ravel('F')


def vech(mat):
    # Gets Fortran-order
    return mat.T.take(_triu_indices(len(mat)))


def _tril_indices(n):
    rows, cols = np.tril_indices(n)
    return rows * n + cols


def _triu_indices(n):
    rows, cols = np.triu_indices(n)
    return rows * n + cols


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
    import matplotlib.pyplot as plt
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

def plot_segmentation(C, aspect=1, classes=None):
    """ Plot a segmentation map.
        Inputs:
            * C: a (height, width) numpy array of integers (classes.
            * aspect: aspect ratio of the image.
    """
    import matplotlib.pyplot as plt
    if classes is not None:
        max_C, min_C = np.max(classes), np.min(classes)
    else:
        max_C, min_C = np.max(C), np.min(C) 

    #get discrete colormap
    cmap = plt.get_cmap('RdBu', max_C-min_C+1)
 
    # set limits .5 outside true range
    mat = plt.matshow(C, aspect=aspect, cmap=cmap, vmin=min_C-.5, vmax=max_C+.5)

    #tell the colorbar to tick at integers
    cax = plt.colorbar(mat, ticks=np.arange(min_C,max_C+1))

def pca_and_save_variance(folder, figname, image, nb_components):
    """ A function that centers data and applies PCA. It also saves a figure of the explained variance.
        Inputs:
            * folder: string.
            * figname: string.
            * image: numpy array to save.
    """
    # center pixels
    h, w, _ = image.shape
    image = image.reshape((-1, image.shape[-1]))
    mean = np.mean(image, axis=0)
    image = image - mean
    # check pixels are centered
    assert (np.abs(np.mean(image, axis=0)) < 1e-9).all()

    # apply PCA
    pca = PCA(nb_components)
    image = pca.fit_transform(image)
    # check pixels are still centered
    assert (np.abs(np.mean(image, axis=0)) < 1e-9).all()
    # reshape image
    image = image.reshape((h, w, nb_components))

    # plot and save explained variance 
    plt.plot(np.arange(1, nb_components+1), np.cumsum(pca.explained_variance_ratio_))
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

def save_segmentation(folder, filename, np_array):
    """ A function that saves a numpy array in a folder. The array and the folder are passed as arguments.
        Inputs:
            * folder: string.
            * filename: string.
            * np_array: numpy array to save.
    """
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    np.save(path, np_array)

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

def assign_classes_segmentation_to_gt(C, gt):
    """ A function that assigns the classes of the segmentation to the ground truth.
        Inputs:
            * C: segmented image.
            * gt: ground truth
        Ouput:
            * segmented image with the right classes.
    """
    # import Hungarian algorithm
    from scipy.optimize import linear_sum_assignment
   
    # if class 0 of gt is used for unnotated pixels then we make the classes of C start from 1
    if len(np.unique(gt)) == (len(np.unique(C))+1):
        C = C + 1
        classes = np.unique(gt)[1:]
    elif len(np.unique(gt)) == len(np.unique(C)):
        classes = np.unique(gt)
    else:
        print('Error: wrong number of classes...')
        sys.exit(1)
    assert (classes == np.unique(C)).all()

    nb_classes = len(classes)

    cost_matrix = np.zeros((nb_classes, nb_classes))

    for i, class_gt in enumerate(classes):
        mask = (gt == class_gt)
        nb_pixels = np.sum(mask)
        for j, class_C in enumerate(classes):
            cost = -np.sum(C[mask] == class_C)#/nb_pixels
            if cost != 0:
                print('i', i, 'j', j, 'cost=', cost)
            cost_matrix[i, j] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    if len(np.unique(gt)) == (len(np.unique(C))+1):
        row_ind += 1
        col_ind += 1
    
    print('row_ind', row_ind)
    print('col_ind', col_ind)
    
    new_C = np.zeros(C.shape)
    for i, j in zip(col_ind, row_ind):
        print(i, 'becomes', j)
        new_C[C==i] = j

    return new_C
