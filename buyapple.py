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
    context.train = True;
    context.assets = [symbol('AAPL'), sid(25)]

def before_trading_start(context, data):
    if context.train == True:
        print( "TRAIN")
        price_history = data.history(context.assets, "price", 6500, "1d")
        print( price_history )
        print( "TRAINING DONE" )
        '''
        this is where we will train the network.
        hopefully it wont time out.
        Get historical data, split it into training, validation.

        '''
    context.train = False

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


def _test_args():
    """Extra arguments to use when zipline's automated tests run this example.
    """
    import pandas as pd

    return {
        'start': pd.Timestamp('2014-01-01', tz='utc'),
        'end': pd.Timestamp('2014-11-01', tz='utc'),
    }









class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff


def example_0():
    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 50
    concat_len = x_dim + mem_cell_ct
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)
    y_list = [-0.5,0.2,0.1, -0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for cur_iter in range(100):
        print( "cur iter: ", cur_iter )
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
            print( "y_pred[%d] : %f" % (ind, lstm_net.lstm_node_list[ind].state.h[0]) )

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print( "loss: ", loss )
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

if __name__ == "__main__":
    example_0()
