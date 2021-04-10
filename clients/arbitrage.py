#!/usr/bin/env python

from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import math
import re

import asyncio
import random

from typing import Optional

"""Constant listed from case packet"""
DAYS_IN_YEAR = 252
LAST_RATE_ROR_USD = 0.25
LAST_RATE_HAP_USD = 0.5
LAST_RATE_HAP_ROR = 2

###### CONSTANTS ####
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
#H, M, U, Z are Month Codes
#6R: ROR/USD
#6H: HAP/USD
#RH: HAP/ROR
contracts = ["6R", "6H", "RH"]
times =  [ "M", "U", "Z"]

FUTURES = [i+j for i in contracts for j in times]
## ALGO #############

def E(book: list) -> float:
    total = sum([lvl.qty for lvl in book])
    ev = 0
    for ask in book:
        ev += float(ask.px) * ((ask.qty)/total)
    return ev

def VAR(book:list, EV: float) -> float:
    V = 0 
    for item in book:
       V += item.qty * (float(item.px) - EV)**2
    return V



'''Rounds price to nearest tick_number above'''
def round_nearest(x, tick=0.0001):
    return round(round(x / tick) * tick, -int(math.floor(math.log10(tick))))

'''Finds daily interest rates from annual rate'''
def daily_rate(daily_rate):
    return math.pow(daily_rate, 1/252)

class ArbitrageBot(UTCBot):
    async def check_arbitrage(self, update: pb.MarketSnapshotMessage):
        for t, asset in update.books.items():
            EV_ask = E(asset.asks)
            EV_bid = E(asset.bids)
            if (len(asset.bids) > 0):
                min_bid = float(min(asset.bids, key = lambda i: i.px).px)
            else:
                min_bid = 0
            self.data[t] = {
                'exp_bid': EV_bid,
                'stdDev_bid': VAR(asset.bids, EV_bid), 
                'exp_ask': EV_ask,
                'stdDev_ask': VAR(asset.asks, EV_ask), 
                'min_bid': min_bid
            }
        #PATH A
        for time in times:
            asset = 'RH' + time
            try:
                path_a = (self.data[asset]['min_bid']) * \
                           (self.data[asset]['exp_bid'])  *  \
                          (self.data['RORUSD']['exp_bid'])
            except ZeroDivisionError:
                path_a = 0
            
            
            # print(f"Path A : {path_a}")
            if path_a > 1 and not self.arbit_opening:
                hap = await self.place_order(asset, pb.OrderSpecType.LIMIT, pb.OrderSpecSide.ASK, 100, round(self.data[asset]['min_bid'], 4))
                endhap = await self.place_order(asset, pb.OrderSpecType.LIMIT, pb.OrderSpecSide.BID, 100, round(self.data[asset]['exp_ask'], 4))
                rorusd = await self.place_order('RORUSD', pb.OrderSpecType.LIMIT, pb.OrderSpecSide.BID, 100, round(self.data['RORUSD']['exp_bid'] - self.data['RORUSD']['stdDev_bid'], 5))
                self.arbit_opening = True

    
    def update_pnl(self, update: pb.PnLMessage):
        pass
    
    
    def update_portfolio(self, update: pb.FillMessage):
        pass


    async def handle_round_started(self):
        """
        Important variables below, some can be more dynamic to improve your case.
        Others are important to tracking pnl - cash, pos, 
        Bidorderid, askorderid track order information so we can modify existing
        orders using the basic MM information (Right now only place 2 bids/2 asks max)
        """
        self.cash = 0.0
        self.pos = {asset:0 for asset in FUTURES + ["RORUSD"]}
        # self.fair = {asset:5 for asset in FUTURES + ["RORUSD"]}
        self.mid = {asset: None for asset in FUTURES + ["RORUSD"]}
        self.max_widths = {asset:0.005 for asset in FUTURES}
        self.arbit_opening = False 
        # self.bidorderid = {asset:["",""] for asset in FUTURES}
        # self.askorderid = {asset:["",""] for asset in FUTURES}


        stored_data = {'exp_bid':0, 'stdDev_bid': 0, 'exp_ask':0, 'stdDev_ask':0}
        self.data = dict([(i, stored_data) for i in FUTURES + ['RORUSD']])



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
 

    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")
        # print('*'*50)
        if kind == "market_snapshot_msg":
            await self.check_arbitrage(update.market_snapshot_msg)
        if kind == "pnl_msg":
            my_m2m = self.cash
            for asset in (FUTURES + ["RORUSD"]):
               my_m2m += self.mid[asset] * self.pos[asset] if self.mid[asset] is not None else 0
            print("M2M", update.pnl_msg.m2m_pnl, my_m2m)
        #Update position upon fill messages of your trades
        if kind == "fill_msg":                
            self.arbit_opening = False
            if update.fill_msg.order_side == pb.FillMessageSide.BUY:
                print(f"BUY: {update.fill_msg.asset}")
                self.cash -= update.fill_msg.filled_qty * float(update.fill_msg.price)
                self.pos[update.fill_msg.asset] += update.fill_msg.filled_qty
            else:
            
                print(f"SELL: {update.fill_msg.asset}")
                self.cash += update.fill_msg.filled_qty * float(update.fill_msg.price)
                self.pos[update.fill_msg.asset] -= update.fill_msg.filled_qty


        if kind =="request_failed_msg":
            pass
        if kind == "liqidation_msg":
            pass
        if kind == "generic_msg":
            pass
            
       
        
if __name__ == "__main__":
    start_bot(ArbitrageBot)