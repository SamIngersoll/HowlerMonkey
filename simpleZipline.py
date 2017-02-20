import tensorflow as tf
from zipline.api import order, record, symbol, sid
from tensorboard import Tensorboard

def initialize(context):
    context.tensorboard = Tensorboard()

def handle_data(context, data):
    #order(symbol('AAPL'), 10)
    #record(AAPL=data.current(symbol('AAPL'), 'price'))
    print( data.current(symbol('AAPL'), 'price'))
    print( context.get_datetime() )
    context.tensorboard.log( 'price', \
                              data.current(symbol('AAPL'), 'price'), \
                              context.get_datetime().date() )
    # tf.summary.scalar('price', data.current(symbol('AAPL'), 'price'))


