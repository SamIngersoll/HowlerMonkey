#   ----------> python -m zipline run -f buyapple.py --start 2015-10-16 --end 2016-7-30 <----------   #



#!/usr/bin/env python
#
# Copyright 2014 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from zipline.api import order, record, symbol, sid
import tensorflow
import tkinter
import numpy as np
#from lstm import LstmParam, LstmNetwork
import lstm

def initialize(context):
    context.assets = [symbol('AAPL'), sid(25)]

# def before_trading_start(context, data):
#     price_history = data.history(context.assets, "price", 6500, "1d")
#     print( price_history )

def handle_data(context, data):
    order(symbol('AAPL'), 10)
    record(AAPL=data.current(symbol('AAPL'), 'price'))


# Note: this function can be removed if running
# this algorithm on quantopian.com
def analyze(context=None, results=None):
    import matplotlib
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    # Plot the portfolio and asset data.
    ax1 = plt.subplot(211)
    results.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('Portfolio value (USD)')
    ax2 = plt.subplot(212, sharex=ax1)
    results.AAPL.plot(ax=ax2)
    ax2.set_ylabel('AAPL price (USD)')

    # Show the plot.
    # plt.gcf().set_size_inches(18, 8)
    plt.show()
