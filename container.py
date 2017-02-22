import sys
import tensorflow as tf
from zipline.api import order, record, symbol, symbols, sid
from filewriter import FileWriter 
from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_from_yahoo, load_bars_from_yahoo
from datetime import datetime, timedelta
import pytz
import os
import time
from network_ import Network
from monsterurl import get_monster
import csv

'''STOCKS = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 
          'CVX', 'DD', 'DIS', 'GE', 'GS', 'HD', 
          'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 
          'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 
          'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WMT', 'XOM']
'''

def initialize( algo ):
    #print( "-------------Initialize-------------" )
    algo.stocks = symbols(*algo.stocks)
    #print( algo.log_dir)
    algo.day = 0
    algo.filewriters = []
    algo.text_file_data = []
    #if algo.log_dir:
    for i in range(len(algo.stocks)):
        algo.filewriters.append( FileWriter(log_dir=algo.log_dir+"/"+str(algo.stocks[i].symbol)) )
    #else:
    #    algo.filewriter = None    
    algo.fields = ["price","open","close","high","low"]
    algo.columns = len(algo.fields)
    #algo.network.train()
    for i in range(len(algo.stocks)):
        for j in range(len(algo.fields)):    
            algo.text_file_data.append(str(algo.stocks[i].symbol)+"_"+algo.fields[j])

def handle_data( algo, data):
    algo.day += 1
    price_history = []
    if ( algo.day >= 3 ):

        #print( "-------------Handle Data-------------" )
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
        print(price_history)
        #prev_bar = list(price_history[i][0] for i in range(len(price_history)))
        #curr_bar = list(price_history[i][0] for i in range(len(price_history)))

def analyze( algo, results ):
    algo.filewriter.writer.close()
    with open(algo.input_data_dir+"/data.csv", 'w') as csvfile:
        algo.writer = csv.writer(csvfile)
        writer.writerows(algo.text_file_data)

class Container:
    def __init__( self, learning_rate, max_steps, hidden1, hidden2, batch_size, lookback=2, input_data_dir=None, log_dir=None, stocks=['AAPL','CAT','NVDA'], fields=[0,0,0], start=datetime(2000,1,1,0,0,0,0,pytz.utc), end=datetime(2016,1,1,0,0,0,0,pytz.utc), liveday=datetime(2011,1,1,0,0,0,0,pytz.utc), individual_name=get_monster(), generation_number=0 ):
        self.log_dir = log_dir
        self.input_data_dir = input_data_dir
        self.individual_name = individual_name
        self.generation_number = generation_number
        self.stocks = stocks     # ['AAPL','CAT','NVDA']
        self.fields = fields
        self.start = start  # datetime(2011, 1, 1, 0, 0, 0, 0, pytz.utc)
        self.end = end      # datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)
        self.classes = 2
        if self.log_dir is None:
            self.log_dir = os.getcwd()+"/logs/"+time.strftime('%d_%m_%Y-%H_%M_%S',time.gmtime())+"/"+str(self.generation_number)+"/"+self.individual_name
        if self.input_data_dir is None:
            self.input_data_dir = self.log_dir
        # Load price data from yahoo.
        data = load_bars_from_yahoo(stocks=self.stocks, indexes={},\
                                 start=start, end=end, adjusted = True)
        data = data.dropna()
        # Create and run the algorithm.
        algorithm = TradingAlgorithm(handle_data=handle_data,\
                                 initialize=initialize) #,\
                                 #identifiers=STOCKS)
        algorithm.stocks = self.stocks
        algorithm.fields = self.fields
        algorithm.start = start
        algorithm.end = end
        algorithm.liveday = liveday
        algorithm.input_data_dir = self.input_data_dir
        algorithm.log_dir = self.log_dir # os.getcwd()+'/logs/'+time.strftime('%d_%m_%Y-%H_%M_%S',time.gmtime())+' = %.2f' % eps
        #algorithm.network = Network( learning_rate, max_steps, hidden1, hidden2, batch_size, self.input_data_dir, self.log_dir, (liveday-start).days, len(self.stocks), self.classes)
        algorithm.network = Network( max_steps = 3 )
        print( self.log_dir )
        results = algorithm.run(data)
