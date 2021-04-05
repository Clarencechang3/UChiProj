# HAR and GARCH forecast volatilty, require implied volatility to compute future value

# Implement GARCH, HAR, SABR here using required paramters

import numpy as np
import scipy


class garch(object):
    def __init__(self, returns):
        """
        To get returns, simply find log(asset[OPEN]) - log(asset[CLOSE])
        for a specified asset.
        """
        self.returns = returns * 100
        self.variance = self.garch_filter(self.garch_optimization())
        self.coeffs = self.garch_optimization()

    def garch_filter(self, params):
        """
        Function returning the variance expression for a GARCH process
        """

        omega = params[0]
        alpha = params[1]
        beta = params[2]

        length = len(self.returns)

        variance = np.zeros(length)

        for i in range(length):
            if i == 0:
                variance[i] = omega / (1 - alpha - beta)
            else:
                variance[i] = (
                    omega + alpha * self.returns[i - 1] ** 2 + beta * variance[i - 1]
                )

        return variance

    def garch_log_likelihood(self, parameters):
        sigma_2 = self.garch_filter(parameters)

        return -np.sum(-np.log(sigma_2) - self.returns ** 2 / sigma_2)

    def garch_optimization(self):
        parameters = [0.1, 0.05, 0.92]

        opt = scipy.optimize.minimize(
            self.garch_log_likelihood,
            parameters,
            bounds=((0.001, 1), (0.001, 1), (0.001, 1)),
        )

        variance = 0.1 ** 2 + opt.x[0] / (1 - opt.x[1] - opt.x[2])

        return np.append(opt.x, variance)


class har(object):
    def __init__(self):
        pass