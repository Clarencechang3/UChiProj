import numpy as np
from numpy.linalg import inv
import pandas as pd
import scipy


# List of valid assets in portfolio
assets = [
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
]


n_assets = len(assets)

# Global storage of price data
returns = pd.read_csv("Case3HistoricalPrices.csv", index_col=0)

# Uses 180-day volatility weighting
risk = np.asarray(returns.iloc[-180:].std())
weighted = np.divide(np.repeat(1, n_assets), risk)
total = np.sum(weighted)

# Ensure weights add to 1
initial_weights = weighted / total


def allocate_portfolio(asset_prices):
    # Add any new pricing information passed into function
    returns.append(asset_prices)

    weights = np.repeat(1 / n_assets, n_assets)
    return weights