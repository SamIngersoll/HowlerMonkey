import sys
import tensorflow as tf
from zipline.api import order, record, symbol, symbols, sid
from tensorboard import FileWriter 
from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_from_yahoo, load_bars_from_yahoo
from datetime import datetime, timedelta
import pytz
import os
import time

'''STOCKS = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 
          'CVX', 'DD', 'DIS', 'GE', 'GS', 'HD', 
          'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 
          'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 
          'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WMT', 'XOM']
'''

def initialize(algo ):
    
    #print( '\n\n\n'+', '.join(str(x) for x in STOCKS)+'\n\n')
    #print( "-------------Initialize-------------" )
    algo.stocks = symbols(*STOCKS)
    #print( algo.stocks[0])
    algo.day = 0
    algo.filewriters = []
    if algo.log_dir:
        for i in range(len(algo.stocks)):
            algo.filewriters.append( FileWriter(log_dir=algo.log_dir+STOCKS[i]) )
    else:
        algo.filewriter = None    
    #algo.sym = symbol('AAPL')
    #algo.order_target = 10
    algo.fields = ["price","open","close","high","low"]
    #algo.stocks

def handle_data( algo, data):
    algo.day += 1
    price_history = []
    if ( algo.day >= 3 ):

        #price_history = data.history(algo.stocks, algo.fields, bar_count=2, frequency="1d")
        #print( "-------------Handle Data-------------" )
        #print(type(symbol('AAPL')))
        #print(price_history.major_xs(price_history.major_axis[0])[algo.fields[0]][algo.stocks[0]])
        #for f in algo.fields:
        for i in range(len(algo.stocks)):
            price_history.append( data.history(algo.stocks[i], algo.fields,\
                                     bar_count=2, frequency="1d").values.tolist())
 
            algo.filewriters[i].log( 'price',\
                                      price_history[i][1][0],\
                                      algo.get_datetime().date() )
            algo.filewriters[i].log( 'change',\
                                      price_history[i][1][3]-\
                                      price_history[i][1][0],\
                                      algo.get_datetime().date() )
            algo.filewriters[i].writer.flush()
        prev_bar = list(price_history[i][0] for i in range(len(price_history)))
        curr_bar = list(price_history[i][0] for i in range(len(price_history)))
                #if curr_bar > prev_bar:
                #order(s, algo.order_target)    
        #algo.filewriter.log( 'price', \
        #                      price_history["price"][0], \
        #                      algo.get_datetime().date() )
        #algo.filewriter.log( 'change', \
        #                      (price_history["close"][-1]-\
        #                        price_history["open"][-1]), \
        #                        algo.get_datetime().date() )
        #algo.filewriters[:].writer.flush()

def analyze( algo, results ):
    algo.filewriter.writer.close()

if __name__ == '__main__':
    STOCKS = ['AAPL','CAT','NVDA']
    start = datetime(2011, 1, 1, 0, 0, 0, 0, pytz.utc)
    loadstart = start - timedelta(days=6)
    end = datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)

    # Load price data from yahoo.
    data = load_bars_from_yahoo(stocks=STOCKS, indexes={}, start=loadstart, end=end, adjusted = True)
    data = data.dropna()
    print( data )
    for eps in [1.0, 1.25, 1.5]:
        print( "HOLA"+'\n'+'\n' )
        # Create and run the algorithm.
        olmar = TradingAlgorithm(handle_data=handle_data,\
                                 initialize=initialize) #,\
                                 #identifiers=STOCKS)
        olmar.eps = eps
        olmar.log_dir = os.getcwd()+'/logs/'+time.strftime('%d_%m_%Y-%H_%M_%S',time.gmtime())+' = %.2f' % eps
        
        # print '-'*100
        print( olmar.log_dir )

        results = olmar.run(data)

