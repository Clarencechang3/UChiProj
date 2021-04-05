import volatility
import numpy as np


def blend(log_returns: object) -> float:
    """
    Blends the volatility estimates of the selected volatility models.
    Returns the estimated volatility for the asset.
    """
    returns = volatility.garch(log_returns)

    # convert variance to volatility and return value
    return np.sqrt(returns.variance)