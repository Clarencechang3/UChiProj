# HAR and GARCH forecast volatilty, require implied volatility to compute future value

# Implement GARCH, HAR, SABR here using required paramters

import numpy as np
import scipy


class garman_klass_volatility:
    def __init__(
        self,
        n: int,
        highs: list,
        lows: list,
        closing_prices: list,
        opening_prices: list,
    ):
        self.n = n
        self.highs = np.array(highs)
        self.lows = np.array(lows)
        self.close = np.array(closing_prices)
        self.open = np.array(opening_prices)

    # calculates the garman_klass_volatility values of input
    def calc_vol(self):
        w = 251 / self.n
        vol = np.power(
            w
            * (
                0.5 * np.power(np.log(np.divide(self.highs, self.lows)), 2)
                - (
                    (2 * np.log(2) - 1)
                    * np.power(np.log(np.divide(self.close, self.open))),
                    2,
                )
            ),
            0.5,
        )

        # returns np array of volatility values
        return vol