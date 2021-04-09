import py_vollib
from py_vollib.black_scholes.implied_volatility import implied_volatility


def implied_volatility_calc(
    price, underlying_price, strike, time_to_expiry, flag
) -> float:
    """
    This function uses the black scholes model to compute the implied volatility
    of the provided asset and parameters as specified
    """
    return implied_volatility(
        price, underlying_price, strike, time_to_expiry / 251, 0, flag.lower()
    )
