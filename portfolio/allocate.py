
import numpy as np
from numpy.linalg import inv
import pandas as pd
from scipy.optimize import newton



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
historical_returns = pd.read_csv("portfolio/Case3HistoricalPrices.csv", index_col=0)

def allocate_portfolio(asset_prices):
    # Add any new pricing information passed into function
    historical_returns.append(asset_prices)
    # Uses 180-day volatility weighting
    risk = np.asarray(historical_returns.iloc[-180:].std())
    weighted = np.divide(np.repeat(1, n_assets), risk)
    total = np.sum(weighted)

    # Ensure weights add to 1
    market_weights = weighted / total

    #covariance matrix of returns, since risk free rate is 0, don't need to calculate excess returns
    cov = historical_returns.cov().to_numpy()

    #return and variance of the market portfolio
    global_return = historical_returns.mean().multiply(market_weights).sum()
    market_var = np.matmul(market_weights.reshape(len(market_weights)).T,
                                        np.matmul(cov, market_weights.reshape(len(market_weights))))
    risk_aversion = global_return / market_var

    calculate_implied_returns = lambda weights : risk_aversion * cov.dot(weights)
    # reg_implied_returns = calculate_implied_returns(market_weights)

    implied_returns_train = lambda weights : (risk_aversion * cov.dot(weights))**2
    optim_weights = newton(implied_returns_train, market_weights) 
    implied_returns = calculate_implied_returns(optim_weights)

    n_views = 3

    P = np.array([
        market_weights.tolist(),#Market Cap
        (1 - market_weights).tolist(),#1 - Market Cap
        historical_returns.var().to_list()
    ])

    def error_cov_matrix(sigma, tau, P):
        matrix = np.diag(np.diag(P.dot(tau * cov).dot(P.T)))
        return matrix
    tau = 0.015
    omega = error_cov_matrix(cov, tau, P)
    scaled_cov = cov * tau

    train_returns = lambda Q_arg : implied_returns + \
                scaled_cov.dot(P.T).dot(inv(P.dot(scaled_cov).dot(P.T) + omega).dot(Q_arg - P.dot(implied_returns)))


    def deriv(Q_arg):
        h = 0.001
        return (train_returns(Q_arg + h) - train_returns(Q_arg - h))/(2*h)

    Q = np.array([1.5, -0.5, -1])
    # Q_optim = newton(train_returns, Q, deriv)

    BL_returns = train_returns(Q)

    BL_weights = inv(cov.values).dot(BL_returns)
    BL_weights = BL_weights/sum(BL_weights)
    return BL_weights


