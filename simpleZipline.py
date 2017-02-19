
import tensorflow as tf
from zipline.api import order, record, symbol, sid
import tensorboard


def initialize(context):
    pass


def handle_data(context, data):
   # order(symbol('AAPL'), 10)
    #record(AAPL=data.current(symbol('AAPL'), 'price'))
    #print( data.current(symbol('AAPL'), 'price'))
    tf.summary.scalar('price', data.current(symbol('AAPL'), 'price'))


if __name__ == '__main__':
    pass
