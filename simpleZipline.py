import tensorflow as tf
from zipline.api import order, record, symbol, sid
from tensorboard import Tensorboard


def initialize(context):
    context.tensorboard = Tensorboard()
    context.sym = symbol('AAPL')
    context.order_target = 10
    context.fields = ["price","open","close","high","low"]

def handle_data(context, data):
    price_history = data.history(context.sym, context.fields, bar_count=2, frequency="1d")
    print(price_history)
    for s in context.fields:
        prev_bar = price_history[s][-2]
        curr_bar = price_history[s][-1]
        #if curr_bar > prev_bar:
            #order(s, context.order_target)    
    context.tensorboard.log( 'price', \
                              price_history["price"][-1], \
                              context.get_datetime().date() )
    context.tensorboard.log( 'delta_high', \
                              (price_history["high"][-1]-\
                                price_history["high"][-1]), \
                              context.get_datetime().date() )
    context.tensorboard.writer.flush()

def analyze( context, results ):
    context.tensorboard.writer.close()


