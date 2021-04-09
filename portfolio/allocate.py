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
historical_returns = pd.read_csv("Case3HistoricalPrices.csv", index_col=0)

# Uses 180-day volatility weighting
risk = np.asarray(historical_returns.iloc[-180:].std())
weighted = np.divide(np.repeat(1, n_assets), risk)
total = np.sum(weighted)

# Ensure weights add to 1
market_weights = weighted / total

#covariance matrix of returns, since risk free rate is 0, don't need to calculate excess returns
cov = historical_returns.cov()

#return and variance of the market portfolio
global_return = historical_returns.mean().multiply(market_weights['weights'].values).sum()
global_variance = np.matmul(market_weights.values.T, np.matmul(cov.values, market_weights.values))

risk_aversion = global_return / global_variance

#implied equilibrium returns of all assets
def implied_rets(risk_aversion, cov_matrix, weights):
    returns = risk_aversion * cov_matrix.dot(weights)
    return returns

implied_returns = implied_rets(risk_aversion, cov, market_weights)
#Array of views
Q = np.array() 
#P represents the matrix identifying assets involved in each view
P = np.asarray([[]])

#covariance matrix of errors of our views
#should be a diagonal matrix
def error_cov_matrix(cov_matrix, tau, P):
    matrix = np.diag(np.diag(tau(cov).dot(P.T)))
    return matrix
tau = 1
omega = error_cov_matrix(cov, tau, P)
scaled_cov = cov * tau
BL_returns = implied_returns + scaled_cov.dot(P.T).dot(inv(P.dot(scaled_cov).dot(P.T) + omega).dot(Q - P.dot(implied_returns)))

BL_weights = inv(cov.values).dot(BL_returns)
BL_weights = BL_weights/sum(BL_weights)


def allocate_portfolio(asset_prices):
    # Add any new pricing information passed into function
    historical_returns.append(asset_prices)

    weights = np.repeat(1 / n_assets, n_assets)
    return weights