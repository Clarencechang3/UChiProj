import numpy as np
import math


"""
Implements a basic Geometric Brownian Motion simulation to simulate 
the random continuous motion of a specified equity. 

Recommended parameters:
DRIFT -- Average return over the time period T
VOLATILITY -- Implied volatility or other simplistic approximation
"""


class Call_Payoff:
    def __init__(self, strike):
        self.strike = strike

    def get_payoff(self, stock_price):
        if stock_price > self.strike:
            return stock_price - self.strike
        else:
            return 0


class Put_Payoff:
    def __init__(self, strike):
        self.strike = strike

    def get_payoff(self, stock_price):
        if stock_price < self.strike:
            return self.strike - stock_price
        else:
            return 0


class Brownian:
    def simulate_paths(self):
        while self.T - self.dt > 0:  # While time remains
            dWt = np.random.normal(0, math.sqrt(self.dt))
            dYt = self.drift * self.dt + self.volatiltiy * dWt  # Change in price
            self.current_price += dYt  # Apply price change
            self.prices.append(self.current_price)

            self.T -= self.dt  # Account for time interval

    def __init__(self, price, drift, volatility, dt, T):
        self.current_price = self.initial_price = price
        self.drift = drift
        self.volatiltiy = volatility
        self.dt = dt
        self.T = T
        self.prices = []
        self.simulate_paths()