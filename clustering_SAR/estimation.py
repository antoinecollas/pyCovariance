import numpy as np

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


def tyler_estimator_covariance_normalisedet(𝐗, tol=0.001, iter_max=20, init=None):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        and normalisation by determinant
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
            * init = Initialisation point of the fixed-point, default is identity matrix
        Outputs:
            * 𝚺 = the estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """
    # Initialisation
    (p,N) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    if init is None:
        𝚺 = (1/N)*𝐗@𝐗.conj().T
    else:
        𝚺 = init
    iteration = 0

    τ=np.zeros((p,N))
    # Recursive algorithm
    while (δ>tol) and (iteration<iter_max):
        # Computing expression of Tyler estimator (with matrix multiplication)
        v=np.linalg.inv(np.linalg.cholesky(𝚺))@𝐗
        a=np.mean(v*v.conj(),axis=0)

        τ[0:p,:] = np.sqrt(np.real(a))
        𝐗_bis = 𝐗 / τ
        𝚺_new = (1/N) * 𝐗_bis@𝐗_bis.conj().T

        # Imposing det constraint: det(𝚺) = 1 DOT NOT WORK HERE
        #𝚺 = 𝚺/(np.linalg.det(𝚺)**(1/p))

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')
        iteration = iteration + 1

        # Updating 𝚺
        𝚺 = 𝚺_new
    
    # Imposing det constraint: det(𝚺) = 1
    𝚺 = 𝚺/(np.linalg.det(𝚺)**(1/p))

    # if iteration == iter_max:
    #     warnings.warn('Recursive algorithm did not converge')

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
