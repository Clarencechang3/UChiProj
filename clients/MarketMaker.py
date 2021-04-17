import math
from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import math
import re
import datetime
import asyncio
import random

from typing import Optional

"""Constant listed from case packet"""
DAYS_IN_YEAR = 252
LAST_RATE_ROR_USD = 0.25
LAST_RATE_HAP_USD = 0.5
LAST_RATE_HAP_ROR = 2


TICK_SIZES = {
    '6RH': 0.00001,
    '6RM': 0.00001,
    '6RU': 0.00001,
    '6RZ': 0.00001,
    '6HH': 0.00002,
    '6HM': 0.00002,
    '6HU': 0.00002,
    '6HZ': 0.00002,
    'RHH': 0.0001,
    'RHM': 0.0001,
    'RHU': 0.0001,
    'RHZ': 0.0001,
    "RORUSD": 0.00001
}


FUTURES = [i+j for i in ["6R", "6H", "RH"] for j in ["H", "M", "U", "Z"]]


'''Rounds price to nearest tick_number above'''


def round_nearest(x, tick=0.0001):
    return round(round(x / tick) * tick, -int(math.floor(math.log10(tick))))


'''Finds daily interest rates from annual rate'''


def daily_rate(daily_rate):
    return math.pow(daily_rate, 1/252)



def expected_value(bid_ask):
    num_orders = sum([order.qty for order in bid_ask.bids + bid_ask.asks])
    EV = sum([float(order.px) * (order.qty/num_orders) for order in bid_ask.bids + bid_ask.asks])
    return EV


def variance(bid_ask):
    num_orders = sum([order.qty for order in bid_ask.bids + bid_ask.asks])
    EV = expected_value(bid_ask)
    var = sum([(order.px - EV)**2 for order in bid_ask.bids + bid_ask.asks])/(num_orders - 1)
    return var


class MarketMaker(UTCBot):

    async def handle_round_started(self):
        """
        Important variables below, some can be more dynamic to improve your case.
        Others are important to tracking pnl - cash, pos, 
        Bidorderid, askorderid track order information so we can modify existing
        orders using the basic MM information (Right now only place 2 bids/2 asks max)
        """

        ##################CONSTANTS#############################
        self.risk_aversion = 0.21
        self.kappa = 2.0
        self.n = -0.005
        self.max_orders = 100
        self.max_time = 2520
        ########################################################

        self.cash = 0.0
        self.time_remaining = 0#set param


        self.bids = {tick: [] for tick in FUTURES + ["RORUSD"]}
        self.asks = {tick: [] for tick in FUTURES + ["RORUSD"]}
        self.num_bids = {tick: 0 for tick in FUTURES + ["RORUSD"]}
        self.num_asks = {tick: 0 for tick in FUTURES + ["RORUSD"]}
        self.pos = {asset: 0 for asset in FUTURES + ["RORUSD"]}
        self.mid = {asset: None for asset in FUTURES + ["RORUSD"]}
        self.max_widths = {asset: 0.005 for asset in FUTURES}

        self.bidorderid = {asset: ["", ""] for asset in FUTURES}
        self.askorderid = {asset: ["", ""] for asset in FUTURES}

        """
        Constant params with respect to assets. Modify this is you would like to change
        parameters based on asset
        """
        self.params = {
            "edge": 0.005,
            "limit": 100,
            "size": 10,
            "spot_limit": 10
        }




    async def place_bid(self, ticker, price, quantity):
        phi = self.max_orders if quantity < 0 else math.exp(-self.n * quantity)
        price = round_nearest(price, tick=TICK_SIZES[ticker])
        await self.place_order(ticker, pb.OrderSpecType.LIMIT, pb.OrderSpecType.BID, phi, price)

    async def place_ask(self, ticker, price, quantity):
        phi = self.max_orders if quantity > 0 else math.exp(-self.n * quantity)
        price = round_nearest(price, tick=TICK_SIZES[ticker])
        await self.place_order(ticker, pb.OrderSpecType.LIMIT, pb.OrderSpecType.ASK, phi, price)



    async def update_bid(self, ticker, price, quantity):
        for bid in self.bids[ticker]:
            phi = self.max_orders if quantity < 0 else math.exp(-self.n * quantity)
            price = round_nearest(price, tick=TICK_SIZES[ticker])
            await self.modify_order(bid, ticker, pb.OrderSpecType.LIMIT, pb.OrderSpecType.BID, phi, price)

    async def update_ask(self, ticker, price, quantity):
        for ask in self.asks[ticker]:
            phi = self.max_orders if quantity > 0 else math.exp(-self.n * quantity)
            price = round_nearest(price, tick=TICK_SIZES[ticker])
            await self.modify_order(ask, ticker, pb.OrderSpecType.LIMIT, pb.OrderSpecType.ASK, phi, price)



    def execute(self, asset, price, spread, qty):
        num_orders = self.num_bids[asset] + self.num_asks[asset]
        if num_orders == 0:
            self.place_ask(asset, price, qty)
            self.place_bid(asset, price, qty)
        
        elif num_orders == 1:
            self.update_ask(asset, price, qty)
            self.update_bid(asset, price, qty)
                   
        
        elif num_orders == 2:
            self.update_ask(asset, price, qty)
            self.update_bid(asset, price, qty)







    async def spread(self, bid_ask):
        N_asks = sum([order.qty for order in bid_ask.asks])
        N_bids = sum([order.qty for order in bid_ask.asks])
        price = expected_value(bid_ask) - (N_asks - N_bids) * self.risk_aversion * variance(bid_ask) * self.time_remaining
        spread = self.risk_aversion * variance(bid_ask) * self.time_remaining + math.log(1 + (self.risk_aversion / self.kappa))
        self.execute(bid_ask.asset, price, spread, N_asks - N_bids)

    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")
        if kind == "pnl_msg":
            pass
            my_m2m = self.cash
            for asset in (FUTURES + ["RORUSD"]):
               my_m2m += self.mid[asset] * self.pos[asset] if self.mid[asset] is not None else 0
            print("M2M", update.pnl_msg.m2m_pnl, my_m2m)
        # Update position upon fill messages of your trades
        elif kind == "fill_msg":
            pass
            if update.fill_msg.order_side == pb.FillMessageSide.BUY:
                self.cash -= update.fill_msg.filled_qty * float(update.fill_msg.price)
                self.pos[update.fill_msg.asset] += update.fill_msg.filled_qty
            else:
                self.cash += update.fill_msg.filled_qty * float(update.fill_msg.price)
                self.pos[update.fill_msg.asset] -= update.fill_msg.filled_qty


        elif kind == "market_snapshot_msg":
            
            print(update.market_snapshot_update.timestamp)
            self.time_remaining = self.max_time - int(update.market_snapshot_update.timestamp)
            for _, asset in update.market_snapshot_msg.books.items():
               self.spread(asset)
               
               
            

        elif kind == "generic_msg":
            print(update)

            # print(update.generic_msg.message)
            # for asset in FUTURES:
            #     await self.place_bids(asset)
            #     await self.place_asks(asset)
            # await self.spot_market()


if __name__ == "__main__":
    start_bot(MarketMaker)
