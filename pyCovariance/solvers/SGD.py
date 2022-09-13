import time
from pymanopt.solvers.solver import Solver


class StochasticGradientDescent(Solver):
    """
    Stochastic gradient descent algorithm.
    """

    def __init__(
        self, checkperiod=100, stepsize_init=100, stepsize_type='decay',
        stepsize__lambda=0.001, stepsize_decaysteps=1000,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.checkperiod = checkperiod

        # Stepsize parameters:
        # Stepsize evolution type. Options are 'decay', 'fix' and 'hybrid'.
        # _lambda is a weighting factor.
        # If stepsize_type = 'hybrid', decaysteps states for how many
        # iterations the step size decays before becoming constant.
        self.stepsize_init = stepsize_init
        self.stepsize_type = stepsize_type
        self.stepsize__lambda = stepsize__lambda
        self.stepsize_decaysteps = stepsize_decaysteps

    def solve(self, problem, x=None):
        """
        Perform optimization using stochastic gradient descent.
        Arguments:
            - problem
                Pymanopt problem setup using the Problem class, this must
                have a .manifold attribute specifying the manifold to optimize
                over, as well as a stochastic gradient. A cost function
                has to be defined but since it is only used in the log
                it can be either on a batch or on all data.
            - x=None
                Optional parameter. Starting point on the manifold. If none
                then a starting point will be randomly generated.
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
        """
        checkperiod = self.checkperiod

        # Manifold, gradient and cost
        man = problem.manifold
        gradient = problem.grad
        objective = problem.cost

        # Stepsize parameters
        stepsize_init = self.stepsize_init
        stepsize_type = self.stepsize_type
        stepsize__lambda = self.stepsize__lambda
        stepsize_decaysteps = self.stepsize_decaysteps

        # Verbosity parameters
        verbosity = problem.verbosity

        # If no starting point is specified, generate one at random.
        if x is None:
            x = man.rand()

        # Initialize iteration counter and timer
        iter = 0
        time0 = time.time()

        # Print and log
        cost = objective(x)

        if self._logverbosity >= 2:
            # TODO: remove that extraiterfields ??
            self._start_optlog(extraiterfields=['gradnorm'])
            self._append_optlog(iter, x, cost)

        while True:
            # Calculate new grad
            grad = gradient(x)

            # Descent direction is minus the gradient
            desc_dir = -grad

            alpha = stepsize_init
            if stepsize_type == 'decay':
                alpha /= 1 + stepsize_init*stepsize__lambda*iter
            elif stepsize_type == 'fix':
                alpha = stepsize_init
            elif stepsize_type == 'hybrid':
                tmp = stepsize_init*stepsize__lambda
                if iter < stepsize_decaysteps:
                    alpha /= 1 + tmp*iter
                else:
                    alpha /= 1 + tmp*stepsize_decaysteps
            else:
                s = 'Unkown type of stepsize. Available:'
                s += ' "fix", "decay", and "hybrid".'
                raise ValueError(s)

            if (iter == 0) and (verbosity >= 2):
                print(" iter\t\t   cost val\t stepsize")
                print("%5d\t%+.16e\t%.5e" % (iter, cost, alpha))

            iter = iter + 1

            # Retract gradient on manifold
            x = man.retr(x, alpha * desc_dir)

            if (iter % checkperiod) == 0:
                cost = objective(x)

                stop_reason = self._check_stopping_criterion(
                    time0, stepsize=alpha, iter=iter)

                if self._logverbosity >= 2:
                    self._append_optlog(iter, x, cost)

                if verbosity >= 2:
                    print("%5d\t%+.16e\t%.5e" % (iter, cost, alpha))

                if stop_reason:
                    if verbosity >= 1:
                        print(stop_reason)
                        print('')
                    break

        if self._logverbosity <= 0:
            return x
        else:
            self._stop_optlog(x, objective(x), stop_reason,
                              time0, stepsize=alpha, iter=iter)
            return x, self._optlog
