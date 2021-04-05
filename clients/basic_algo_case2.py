#!/usr/bin/env python

from dataclasses import astuple
from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto

import brownian
import implied_volatility
import volatility
import volatility_blending

import numpy as np
import asyncio
import random
import math

option_strikes = [90, 95, 100, 105, 110]


class Case2ExampleBot(UTCBot):
    """
    An example bot for Case 2 of the 2021 UChicago Trading Competition. We recommend that you start
    by reading through this bot and understanding how it works. Then, make a copy of this file and
    start trying to write your own bot!
    """

    async def handle_round_started(self):
        """
        This function is called when the round is started. You should do your setup here, and
        start any tasks that should be running for the rest of the round.
        """

        # This variable will be a map from asset names to positions. We start out by initializing it
        # to zero for every asset.
        self.positions = {}
        self.trade_prices = {}

        self.positions["UC"] = 0
        for strike in option_strikes:
            for flag in ["C", "P"]:
                self.positions[f"UC{strike}{flag}"] = 0

        # Stores the current day (starting from 0 and ending at 5). This is a floating point number,
        # meaning that it includes information about partial days
        self.current_day = 0

        # Stores the current value of the underlying asset
        self.underlying_price = 100

    def compute_implied_volatility(self) -> float:
        """
        This function estimates the implied volatility of the underlying asset. Utilizes the
        provided py_vollib package and the templated code from Piazza.
        """
        total_vol = 0
        total_weight = 0

        for strike in option_strikes:
            for flag in ["C", "P"]:
                # Weight to be used for the weighted average
                # (for motivation, look up how vega is computed in B-S model)
                weight = math.exp(0.5 * math.log(self.underlying_price / strike) ** 2)

                # This function should compute the Black-Scholes implied volatility
                vol = implied_volatility.implied_volatility_calc(
                    price[f"UC{strike}{flag}"],
                    self.underlying_price,
                    strike,
                    time_to_expiry,
                    flag,
                )

                total_vol += weight * vol
                total_weight += weight

        exchange_vol_estimate = total_vol / total_weight

    def compute_vol_estimate(self) -> float:
        """
        This function is used to provide an estimate of underlying's volatility. Because this is
        an example bot, we just use a placeholder value here. We recommend that you look into
        different ways of finding what the true volatility of the underlying is.
        """

        # calculate log_returns here; use np array?
        log_returns = []

        return volatility_blending.blend(log_returns)

    # TODO: Research various pricing models
    # Takes in volatility calculated in volatility.py
    def compute_options_price(
        self,
        flag: str,
        underlying_px: float,
        strike_px: float,
        time_to_expiry: float,
        volatility: float,
    ) -> float:
        """
        This function should compute the price of an option given the provided parameters. Some
        important questions you may want to think about are:
            - What are the units associated with each of these quantities?
            - What formula should you use to compute the price of the option?
            - Are there tricks you can use to do this more quickly?
        You may want to look into the py_vollib library, which is installed by default in your
        virtual environment.
        """
        paths = 50
        drift = 0.08
        dt = 1 / 251
        T = 1
        price_paths = []

        for i in range(0, paths):
            price_paths.append(
                brownian.Brownian(underlying_px, drift, volatility, dt, T).prices
            )

        call_payoffs = []
        ec = brownian.Call_Payoff(strike_px)

        for price_path in price_paths:
            call_payoffs.append(ec.get_payoff(price_path[-1]))

        # * 100 since options are in blocks of 100
        return np.average(call_payoffs) * 100

    async def update_options_quotes(self):
        """
        This function will update the quotes that the bot has currently put into the market.

        In this example bot, the bot won't bother pulling old quotes, and will instead just set new
        quotes at the new theoretical price every time a price update happens. We don't recommend
        that you do this in the actual competition
        """
        # TODO: What should this value actually be?
        time_to_expiry = 21 / 252
        vol = self.compute_vol_estimate()

        for strike in option_strikes:
            for flag in ["C", "P"]:
                asset_name = f"UC{strike}{flag}"
                theo = self.compute_options_price(
                    flag, self.underlying_price, strike, time_to_expiry, vol
                )

                bid_response = await self.place_order(
                    asset_name,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.BID,
                    1,  # How should this quantity be chosen?
                    theo - 0.30,  # How should this price be chosen?
                )
                assert bid_response.ok

                ask_response = await self.place_order(
                    asset_name,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.ASK,
                    1,
                    theo + 0.30,
                )
                assert ask_response.ok

    def calc_price_helper(self, book: object) -> float:
        if len(book) < 3:
            bid = ask = 0
            quantity = 0
            # find weighted bid price
            for i in range(len(book)):
                quantity += float(book.bids[i].qty)
                bid += float(book.bids[i].px) * float(book.bids[i].qty)

            bid /= quantity

            quantity = 0
            # find weighted ask price
            for i in range(len(book)):
                quantity += float(book.asks[i].qty)
                ask += float(book.asks[i].px) * float(book.asks[i].qty)

            ask /= quantity

            return (bid, ask)

        else:
            bid = ask = 0
            quantity = 0
            # find weighted bid price
            for i in range(3):
                quantity += float(book.bids[i].qty)
                bid += float(book.bids[i].px) * float(book.bids[i].qty)

            bid /= quantity

            quantity = 0
            # find weighted ask price
            for i in range(3):
                quantity += float(book.asks[i].qty)
                ask += float(book.asks[i].px) * float(book.asks[i].qty)

            ask /= quantity

            return bid, ask

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

            # Compute the mid price of the market and store it
            self.underlying_price = (
                float(book.bids[0].px) + float(book.asks[0].px)
            ) / 2

            for strike in option_strikes:
                for flag in ["C", "P"]:
                    asset_name = f"UC{strike}{flag}"
                    book_ = update.market_snapshot_msg.books[asset_name]

                    bid, ask = self.calc_price_helper(book_)

                    self.trade_prices[asset_name] = (bid + ask) / 2

            await self.update_options_quotes()

        elif (
            kind == "generic_msg"
            and update.generic_msg.event_type == pb.GenericMessageType.MESSAGE
        ):
            # The platform will regularly send out what day it currently is (starting from day 0 at
            # the start of the case)
            self.current_day = float(update.generic_msg.message)

        elif kind == "trade_msg":
            # There are other pieces of information the exchange provides feeds for. See if you can
            # find ways to use them to your advantage (especially when more than one competitor is
            # in the market)
            pass


if __name__ == "__main__":
    start_bot(Case2ExampleBot)