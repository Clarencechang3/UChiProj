import volatility
import numpy as np
import arch


def blend(returns, last_volatility) -> float:
    """
    Blends the volatility estimates of the selected volatility models.
    Returns the estimated volatility for the asset.
    """
    # if it's the first day, don't want to volatility blend as this will be highly inaccurate
    # Instead, use a pre-computed volatility (average volatility of sample paths)
    if len(returns) < 200:
        return 1.6879372939912174

    alpha_1 = 0.4
    garch = arch.arch_model(returns, vol="garch", p=1, o=0, q=1)
    garch_fitted = garch.fit()
    g_pred = garch_fitted.forecast()

    alpha_2 = 0.2
    figarch = arch.arch_model(returns, vol="figarch", p=1)
    figarch_fitted = figarch.fit()
    f_pred = figarch_fitted.forecast()

    alpha_3 = 0.2
    egarch = arch.arch_model(returns, vol="egarch", p=1, o=1, q=1)
    egarch_fitted = egarch.fit()
    e_pred = egarch_fitted.forecast()

    alpha_4 = 0.2

    # convert variance to volatility and return value
    return (
        alpha_1 * np.sqrt(g_pred.variance)
        + alpha_2 * np.sqrt(f_pred.variance)
        + alpha_3 * np.sqrt(e_pred.variance)
        + alpha_4 * last_volatility
    )
