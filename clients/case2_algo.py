#!/usr/bin/env python

from dataclasses import astuple
from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto

import arch
import py_vollib
from py_vollib.black_scholes.implied_volatility import implied_volatility

import numpy as np
import asyncio
import random
import math
from scipy import stats

# ---------------------------------------BINOMIAL MODEL---------------------------------------
def binomial_model(N, S0, u, r, K):
    """
    N = number of binomial iterations
    S0 = initial stock price
    u = factor change of upstate
    r = risk free interest rate per annum
    K = strike price
    """
    d = 1 / u
    p = (1 + r - d) / (u - d)
    q = 1 - p

    # make stock price tree
    stock = np.zeros([N + 1, N + 1])
    for i in range(N + 1):
        for j in range(i + 1):
            stock[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # Generate option prices recursively
    option = np.zeros([N + 1, N + 1])
    option[:, N] = np.maximum(np.zeros(N + 1), (stock[:, N] - K))
    for i in range(N - 1, -1, -1):
        for j in range(0, i + 1):
            option[j, i] = (
                1 / (1 + r) * (p * option[j, i + 1] + q * option[j + 1, i + 1])
            )
    return option


# ---------------------------------------/BINOMIAL MODEL---------------------------------------

# ---------------------------------------BROWNIAN MOTION---------------------------------------


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


# ---------------------------------------/BROWNIAN MOTION---------------------------------------

# ---------------------------------------VOL BLENDING---------------------------------------


def blend(returns, last_implied, pct_return, time) -> float:
    """
    Blends the volatility estimates of the selected volatility models.
    Returns the estimated volatility for the asset.
    """
    # if it's the first day, don't want to volatility blend as this will be highly inaccurate
    # Instead, use a pre-computed volatility (average volatility of sample paths)
    # if len(returns) < 200:
    #     return 1.6879372939912174

    if len(returns) > 6:
        returns = returns[-6:]

    garch = arch.arch_model(returns, mean="HAR")
    res = garch.fit(update_freq=0)

    omega = res.params["omega"]
    beta = res.params["beta[1]"]
    alpha = res.params["alpha[1]"]

    return (omega + alpha * pct_return + beta * last_implied) ** 0.5


# ---------------------------------------/VOL BLENDING---------------------------------------

# ---------------------------------------GARMAN KLASS---------------------------------------
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


# ---------------------------------------/GARMAN KLASS---------------------------------------
delta_threshold = 2000
gamma_threshold = 20
theta_threshold = -25
vega_threshold = 35
option_strikes = [90, 95, 100, 105, 110]
historical_vol = []


class derivative:
    def d1_calc(self, price, strike, time_to_expiry, vol):
        T = time_to_expiry / 251
        return (math.log(price / strike) + (vol ** 2) / 2 * T) / (vol * math.sqrt(T))

    def d2_calc(self, d1, vol, time_to_expiry):
        T = time_to_expiry / 251
        return d1 - vol * math.sqrt(T)

    def delta_calc(self, d1):
        if self.flag == "C":
            return 100 * stats.norm.cdf(d1)
        else:
            return 100 * (stats.norm.cdf(d1) - 1)

    def gamma_calc(self, d1, u_price, vol, time_to_expiry):
        T = time_to_expiry / 251
        return stats.norm.pdf(d1) / (u_price * vol * math.sqrt(T))

    def vega_calc(self, d1, u_price, time_to_expiry):
        T = time_to_expiry / 251
        return stats.norm.pdf(d1) * u_price * math.sqrt(T)

    def theta_calc(self, d1, u_price, vol, time_to_expiry):
        T = time_to_expiry / 251
        return -(u_price * stats.norm.pdf(d1) * vol) / (2 * math.sqrt(T))

    def __init__(self, asset_name: str, flag: str, strike: float, vol: float):
        self.time_to_expiry = 26
        self.asset_name = asset_name
        self.price = 100
        self.flag = flag
        self.d1 = 0
        self.d2 = 0
        self.delta = 0
        self.gamma = 0
        self.vega = 0
        self.theta = 0
        self.strike = strike
        self.vol = vol


class Case2Algo(UTCBot):
    def risk_limit_check(self):
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0

        for asset, qty in self.positions.items():
            if asset != "UC" and qty != 0:
                price = self.underlying_price
                strike = self.derivatives[asset].strike
                time_to_expiry = self.derivatives[asset].time_to_expiry
                vol = self.derivatives[asset].vol

                self.derivatives[asset].d1 = self.derivatives[asset].d1_calc(
                    price, strike, time_to_expiry, vol
                )

                self.derivatives[asset].delta = self.derivatives[asset].delta_calc(
                    self.derivatives[asset].d1
                )
                self.derivatives[asset].gamma = self.derivatives[asset].gamma_calc(
                    self.derivatives[asset].d1,
                    price,
                    self.derivatives[asset].vol,
                    time_to_expiry,
                )
                self.derivatives[asset].vega = self.derivatives[asset].vega_calc(
                    self.derivatives[asset].d1, price, time_to_expiry
                )
                self.derivatives[asset].theta = self.derivatives[asset].theta_calc(
                    self.derivatives[asset].d1,
                    price,
                    self.derivatives[asset].vol,
                    time_to_expiry,
                )

                total_delta += self.derivatives[asset].delta
                total_gamma += self.derivatives[asset].gamma
                total_theta += self.derivatives[asset].theta
                total_vega += self.derivatives[asset].vega

        if (
            total_delta > delta_threshold
            or total_gamma > gamma_threshold
            or total_vega > vega_threshold
            or total_delta < -delta_threshold
            or total_gamma < -gamma_threshold
            or total_theta < theta_threshold
            or total_vega < -vega_threshold
        ):
            return True
        else:
            return False

    async def handle_round_started(self):
        """
        This function is called when the round is started. You should do your setup here, and
        start any tasks that should be running for the rest of the round.
        """

        # This variable will be a map from asset names to positions. We start out by initializing it
        # to zero for every asset.
        self.positions = {}
        self.previous_day = 0

        # initially equal to aggregate sample volatility
        self.last_implied = 0.5

        # TODO: optimize strategy according to number of competitors in market; update w/ market updates
        self.num_competitors = 0

        # maps asset name to derivative object
        self.derivatives = {}

        self.positions["UC"] = 0

        for strike in option_strikes:
            for flag in ["C", "P"]:
                asset_name = "UC" + str(strike) + flag
                assert not asset_name in self.positions
                self.positions[asset_name] = 0
                assert not asset_name in self.derivatives

                flag = "C" if (flag == "C") else "P"
                self.derivatives[asset_name] = derivative(
                    asset_name, "test", strike, 0.1
                )

        # Stores the current day (starting from 0 and ending at 5). This is a floating point number,
        # meaning that it includes information about partial days
        # should change this to update whenever we receive a new market update
        self.current_day = 0
        self.updates = 0
        # Stores the starting value of the underlying asset
        self.underlying_price = 100
        self.historical_prices = [100]

    def compute_implied_volatility(self) -> float:
        """
        This function estimates the implied volatility of the option. Utilizes the
        provided py_vollib package and the templated code from Piazza.
        """
        total_vol = 0
        total_weight = 0

        for strike in option_strikes:
            for flag in ["C", "P"]:
                asset_name = "UC" + str(strike) + str(flag)
                # Weight to be used for the weighted average
                weight = math.exp(0.5 * math.log(self.underlying_price / strike) ** 2)

                # computes BS implied volatility using py_vollib
                vol = implied_volatility(
                    self.derivatives[asset_name].price,
                    self.underlying_price,
                    strike,
                    self.derivatives[asset_name].time_to_expiry / 251,
                    0,
                    flag.lower(),
                )

                total_vol += weight * vol
                total_weight += weight

        # returns weighted average of implied volatility
        return total_vol / total_weight

    def compute_vol_estimate(self, longterm: bool) -> float:
        if len(self.historical_prices) <= 5:
            return self.compute_implied_volatility()  # adjust for beginning of round
        else:
            returns = np.asarray(self.historical_prices, dtype=np.float)

        # for longterm purposes, simply use historical volatility
        if longterm:
            # Returns should be length 1000
            assert len(returns) == 1000

            closing_prices = np.take(returns, (199, 399, 599, 799, 999))
            opening_prices = np.take(returns, (0, 200, 400, 600, 800))

            arr_returns = np.reshape(returns, newshape=(200, 5))
            arr_highs = np.max(arr_returns, axis=0)
            arr_lows = np.min(arr_returns, axis=0)

            vol = garman_klass_volatility(
                n=len(self.historical_prices),
                highs=arr_highs,
                lows=arr_lows,
                closing_prices=closing_prices,
                opening_prices=opening_prices,
            )

            return vol

        # for shorter term, blend several useful models to incorporate
        # long and short-term
        else:
            pct_return = self.underlying_price / 100
            vol = 0.3 * blend(returns, self.last_implied, pct_return, self.current_day)
            self.last_implied = self.compute_implied_volatility()

            historical_vol.append(self.last_implied)

            vol += 0.7 * self.last_implied

            return vol

    # vol computed as short-term estimate (compute_vol_estimate)
    def compute_shortterm_options_price(
        self,
        flag: str,
        underlying_px: float,
        strike_px: float,
        time_to_expiry: float,
        volatility: float,
    ) -> float:
        # price using black-scholes

        val = py_vollib.black_scholes.undiscounted_black(
            F=underlying_px,
            K=strike_px,
            sigma=volatility,
            flag=flag.lower(),
            t=time_to_expiry / 251,
        )

        return val

    # Takes in volatility calculated in volatility.py
    def compute_longterm_options_price(
        self,
        flag: str,
        underlying_px: float,
        strike_px: float,
        time_to_expiry: float,
        volatility: float,
    ) -> float:

        # Hyperparameters
        paths = 100
        drift = 0.08
        dt = 1 / 251
        T = time_to_expiry / 251

        price_paths = np.asarray([])

        for _ in range(0, paths):
            price_paths.append(Brownian(underlying_px, drift, volatility, dt, T).prices)

        if flag == "P":
            put_payoffs = []
            ep = Put_Payoff(strike_px)

            for price_path in price_paths:
                put_payoffs.append(ep.get_payoff(price_path[-1]))

            return np.average(put_payoffs) * 100

        else:
            call_payoffs = []
            ec = Call_Payoff(strike_px)

            for price_path in price_paths:
                call_payoffs.append(ec.get_payoff(price_path[-1]))

            return np.average(put_payoffs) * 100

    async def update_options_quotes(self, update):
        """
        This function will update the quotes that the bot has currently put into the market.

        In this example bot, the bot won't bother pulling old quotes, and will instead just set new
        quotes at the new theoretical price every time a price update happens. We don't recommend
        that you do this in the actual competition
        """
        # calculates the volatility of the underlying
        vol = self.compute_vol_estimate(longterm=False)
        self.updates += 1

        baseline_vol = np.median(np.asarray(historical_vol))

        if self.compute_implied_volatility() < baseline_vol * 1.20:
            take = 0.002
        else:
            take = 0.01

        at_limit = self.risk_limit_check()
        if not at_limit:
            for strike in option_strikes:
                for flag in ["C", "P"]:
                    asset_name = "UC" + str(strike) + str(flag)
                    time_to_expiry = self.derivatives[asset_name].time_to_expiry

                    theo = self.compute_shortterm_options_price(
                        flag, self.underlying_price, strike, time_to_expiry, vol
                    )

                    price = self.derivatives[asset_name].price

                    if price * 1.05 <= theo and not price * 1.50 <= theo:
                        qty = int(min(5, (theo - price * 1.05) / 0.2))
                        price = (
                            float(
                                update.market_snapshot_msg.books[asset_name].bids[0].px
                            )
                            + float(
                                update.market_snapshot_msg.books[asset_name].asks[0].px
                            )
                        ) / 2
                        bid_response = await self.place_order(
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.BID,
                            qty,  # How should this quantity be chosen?
                            round(
                                float(
                                    update.market_snapshot_msg.books[asset_name]
                                    .bids[0]
                                    .px
                                )
                                + take * (theo - price),
                                1,
                            ),  # How should this price be chosen?
                        )
                        assert bid_response.ok
                        gamma_hedge = self.gamma_hedge(
                            asset_name, qty, price, strike, time_to_expiry, vol
                        )

                        if gamma_hedge > 0:
                            if self.derivatives[asset_name].flag == "P":
                                asset = (
                                    "UC"
                                    + str(self.derivatives[asset_name].strike)
                                    + "C"
                                )
                            else:
                                asset = (
                                    "UC"
                                    + str(self.derivatives[asset_name].strike)
                                    + "P"
                                )
                            gamma_hedge_response = await self.place_order(
                                asset,
                                pb.OrderSpecType.LIMIT,
                                pb.OrderSpecSide.ASK,
                                int(gamma_hedge),
                                round(self.underlying_price, 1),
                            )
                            assert gamma_hedge_response.ok
                        if gamma_hedge < 0:
                            if self.derivatives[asset_name].flag == "P":
                                asset = (
                                    "UC"
                                    + str(self.derivatives[asset_name].strike)
                                    + "C"
                                )
                            else:
                                asset = (
                                    "UC"
                                    + str(self.derivatives[asset_name].strike)
                                    + "P"
                                )
                            gamma_hedge_response = await self.place_order(
                                asset,
                                pb.OrderSpecType.LIMIT,
                                pb.OrderSpecSide.BID,
                                int(-gamma_hedge),
                                round(self.underlying_price, 1),
                            )
                            assert gamma_hedge_response.ok

                        # total_delta = 0
                        hedge = self.delta_hedge()
                        if hedge > 0:
                            hedge_response = await self.place_order(
                                "UC",
                                pb.OrderSpecType.MARKET,
                                pb.OrderSpecSide.ASK,
                                hedge,
                            )
                            assert hedge_response.ok
                        if hedge < 0:
                            hedge_response = await self.place_order(
                                "UC",
                                pb.OrderSpecType.MARKET,
                                pb.OrderSpecSide.BID,
                                -hedge,
                            )
                            assert hedge_response.ok

                    if price * 0.95 >= theo and not price * 0.7 >= theo:
                        qty = int(min(5, (price * 0.95 - theo) / 0.2))
                        price = (
                            float(
                                update.market_snapshot_msg.books[asset_name].bids[0].px
                            )
                            + float(
                                update.market_snapshot_msg.books[asset_name].asks[0].px
                            )
                        ) / 2
                        ask_response = await self.place_order(
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.ASK,
                            qty,
                            round(
                                float(
                                    update.market_snapshot_msg.books[asset_name]
                                    .asks[0]
                                    .px
                                )
                                - take * (price - theo),
                                1,
                            ),
                        )
                        assert ask_response.ok
                        gamma_hedge = self.gamma_hedge(
                            asset_name, qty, price, strike, time_to_expiry, vol
                        )

                        if gamma_hedge > 0:
                            if self.derivatives[asset_name].flag == "P":
                                asset = (
                                    "UC"
                                    + str(self.derivatives[asset_name].strike)
                                    + "C"
                                )
                            else:
                                asset = (
                                    "UC"
                                    + str(self.derivatives[asset_name].strike)
                                    + "P"
                                )
                            gamma_hedge_response = await self.place_order(
                                asset,
                                pb.OrderSpecType.LIMIT,
                                pb.OrderSpecSide.ASK,
                                int(gamma_hedge),
                                round(self.underlying_price, 1),
                            )
                            assert gamma_hedge_response.ok
                        if gamma_hedge < 0:
                            if self.derivatives[asset_name].flag == "P":
                                asset = (
                                    "UC"
                                    + str(self.derivatives[asset_name].strike)
                                    + "C"
                                )
                            else:
                                asset = (
                                    "UC"
                                    + str(self.derivatives[asset_name].strike)
                                    + "P"
                                )
                            gamma_hedge_response = await self.place_order(
                                asset,
                                pb.OrderSpecType.LIMIT,
                                pb.OrderSpecSide.BID,
                                int(-gamma_hedge),
                                round(self.underlying_price, 1),
                            )
                            assert gamma_hedge_response.ok

                        hedge = self.delta_hedge()
                        if hedge > 0:
                            hedge_response = await self.place_order(
                                "UC",
                                pb.OrderSpecType.MARKET,
                                pb.OrderSpecSide.ASK,
                                hedge,
                            )
                            assert hedge_response.ok
                        if hedge < 0:
                            hedge_response = await self.place_order(
                                "UC",
                                pb.OrderSpecType.MARKET,
                                pb.OrderSpecSide.BID,
                                -hedge,
                            )
                            assert hedge_response.ok
        if at_limit:
            hedge = self.delta_hedge()
            if hedge > 0:
                hedge_response = await self.place_order(
                    "UC",
                    pb.OrderSpecType.MARKET,
                    pb.OrderSpecSide.ASK,
                    hedge,
                )
                assert hedge_response.ok
            if hedge < 0:
                hedge_response = await self.place_order(
                    "UC",
                    pb.OrderSpecType.MARKET,
                    pb.OrderSpecSide.BID,
                    -hedge,
                )
                assert hedge_response.ok

    def delta_hedge(self):

        hedge_amount = 0
        for asset, qty in self.positions.items():
            if asset != "UC" and qty > 0:
                price = self.underlying_price
                strike = self.derivatives[asset].strike
                time_to_expiry = self.derivatives[asset].time_to_expiry
                vol = self.derivatives[asset].vol

                self.derivatives[asset].d1 = self.derivatives[asset].d1_calc(
                    price, strike, time_to_expiry, vol
                )

                self.derivatives[asset].delta = self.derivatives[asset].delta_calc(
                    self.derivatives[asset].d1
                )

                hedge_amount += self.derivatives[asset].delta

        return math.floor(hedge_amount)

    def gamma_hedge(self, asset, qty, price, strike, time_to_expiry, vol) -> float:
        self.derivatives[asset].d1 = self.derivatives[asset].d1_calc(
            price, strike, time_to_expiry, vol
        )
        self.derivatives[asset].gamma = self.derivatives[asset].gamma_calc(
            self.derivatives[asset].d1,
            price,
            self.derivatives[asset].vol,
            time_to_expiry,
        )
        total_gamma = qty * self.derivatives[asset].gamma
        if self.derivatives[asset].flag == "C":
            hedge_asset = "UC" + str(strike) + "P"
        else:
            hedge_asset = "UC" + str(strike) + "C"
        self.derivatives[hedge_asset].d1 = self.derivatives[hedge_asset].d1_calc(
            price, strike, time_to_expiry, vol
        )
        self.derivatives[hedge_asset].gamma = self.derivatives[hedge_asset].gamma_calc(
            self.derivatives[hedge_asset].d1,
            price,
            self.derivatives[hedge_asset].vol,
            time_to_expiry,
        )
        hedge_amount = float(total_gamma) / self.derivatives[hedge_asset].gamma
        return hedge_amount

    def parity_check(self):
        at_parity = []
        for strike in option_strikes:
            call = "UC" + strike + "C"
            put = "UC" + strike + "P"
            if (
                self.derivatives[put].price + self.underlying_price
                != self.derivatives[call].price + strike
            ):
                at_parity.append(False)
            else:
                at_parity.append(True)
        return at_parity

    # checks if put-call parity exists for each strike price
    # returns a list of bools, first spot in list corresponds to K = 90

    def calc_price_helper(self, book: object) -> (float, float):
        return float(book.bids[0].px), float(book.asks[0].px)

    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")

        if kind == "pnl_msg":
            # When you hear from the exchange about your PnL, print it out
            print("My PnL:", update.pnl_msg.m2m_pnl)

        elif kind == "fill_msg":
            # When you hear about a fill you had, update your positions
            fill_msg = update.fill_msg

            if fill_msg.order_side == pb.FillMessageSide.BUY:
                self.positions[fill_msg.asset] += update.fill_msg.filled_qty
            else:
                self.positions[fill_msg.asset] -= update.fill_msg.filled_qty

        elif kind == "market_snapshot_msg":
            # When we receive a snapshot of what's going on in the market, update our information
            # about the underlying price.
            book = update.market_snapshot_msg.books["UC"]

            # new timestep, decrease all time-to-expiry values accordingly (1 / 200 = 0.005)
            for _, derivative_obj in self.derivatives.items():
                derivative_obj.time_to_expiry -= 0.005

            prev_price = self.underlying_price
            # Compute the mid price of the market and store it
            self.underlying_price = (
                float(book.bids[0].px) + float(book.asks[0].px)
            ) / 2

            if float(abs(prev_price - self.underlying_price)) / prev_price > 0.05:
                self.delta_hedge()

            self.historical_prices.append(self.underlying_price)

            for strike in option_strikes:
                for flag in ["C", "P"]:
                    asset_name = "UC" + str(strike) + str(flag)
                    book_ = update.market_snapshot_msg.books[asset_name]

                    bid, ask = self.calc_price_helper(book_)

                    self.derivatives[asset_name].price = (bid + ask) / 2

            await self.update_options_quotes(update)

        elif (
            kind == "generic_msg"
            and update.generic_msg.event_type == pb.GenericMessageType.MESSAGE
        ):
            # The platform will regularly send out what day it currently is (starting from day 0 at
            # the start of the case)
            self.current_day = float(update.generic_msg.message)
            self.current_day_int = math.floor(self.current_day)

            # if last noted day is not this day (we've changed days)
            if self.current_day_int is not self.previous_day:
                self.previous_day = self.current_day_int
                self.delta_hedge()

        elif (
            kind == "generic_msg"
            and update.generic_msg.event_type == pb.GenericMessageType.ROUND_ENDED
        ):
            # TODO: Add logic for offloading at the end of the round

            longterm_prices = {}

            vol = self.compute_vol_estimate(longterm=True)

            for strike in option_strikes:
                for flag in ["C", "P"]:
                    asset_name = "UC" + str(strike) + str(flag)

                    # compute longterm prices
                    longterm_prices[asset_name] = self.compute_longterm_options_price(
                        flag,
                        self.underlying_price,
                        strike,
                        self.derivatives[asset_name].time_to_expiry,
                        vol,
                    )

            # if monte carlo higher than current (or lower), keep
            # TODO: Risk management

        elif kind == "trade_msg":
            # There are other pieces of information the exchange provides feeds for.
            pass


if __name__ == "__main__":
    start_bot(Case2Algo)