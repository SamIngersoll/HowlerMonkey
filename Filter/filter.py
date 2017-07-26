#Source: https://pypi.python.org/pypi/Yahoo-ticker-downloader

import subprocess
import os
import csv
import yahoo_finance
import time

# Downloads all yahoo data based on the flags in the command, to a csv file in the same directory.
def download_sids():
    print("Getting all stock symbols...")
    try:
        process = subprocess.Popen("YahooTickerDownloader.py stocks -m us", stdout=subprocess.PIPE, shell=True)
        process.wait()
        print("\nSuccessfully completed symbol retreival")
    except Exception as e:
        print(e)

# Reads the downloaded file, goes through each sid, checks if it meets our criteria, and returns a list of sids.
#TODO: Need to establish criteria
def filter_sids():
    print("\nFiltering results...")
    sids = []
    with open("{}/stocks.csv".format(os.path.dirname(os.path.realpath(__file__)))) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            sid = row[0]
            try:
                share = yahoo_finance.Share(sid)
                time.sleep(.05)
                pe = share.get_price_earnings_ratio()
                time.sleep(.05)
                dy = share.get_dividend_yield()
                time.sleep(.05)
            except Exception as e:
                print("First Wait...")
                time.sleep(5)
                try:
                    share = yahoo_finance.Share(sid)
                    time.sleep(.05)
                    pe = share.get_price_earnings_ratio()
                    time.sleep(.05)
                    dy = share.get_dividend_yield()
                    time.sleep(.05)
                except Exception as e:
                    print("Second Wait...")
                    time.sleep(10)
                    try:
                        share = yahoo_finance.Share(sid)
                        time.sleep(.05)
                        pe = share.get_price_earnings_ratio()
                        time.sleep(.05)
                        dy = share.get_dividend_yield()
                        time.sleep(.05)
                    except Exception as e:
                        print(e)
            if pe is not None and dy is not None:
                if float(pe) > 5 or float(dy) > 5:
                    print(sid)
                    sids.append(sid)
    print("\nResults filtered\n")
    return sids
