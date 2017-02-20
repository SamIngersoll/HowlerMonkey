
import tensorflow as tf
from zipline.api import order, record, symbol, sid
import tensorboard


def initialize(context):
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

    #tf.summary.scalar('today_price',tday_price) 


#if __name__ == '__main__':
#    pass
