#!/usr/bin/env python2.7
import os
import pandas_datareader as pdr
import yfinance as fix
import dataMiner as D


err_arrsy = []

fix.pdr_override()
def loadfile(ticker_type, ticker, datapath, startdate, enddate):
    """
    :param ticker_type:
    :param ticker:
    :param datapath:
    :param startdate:
    :param enddate:
    :return:
    """

    err_arrsy = []

    if ticker_type == 'en':
        print("Load data from Yahoo, ticker:", ticker)
        print("From %s date" % startdate)
        print("Till %s date" % enddate)

        try:
            __raw_data = pdr.get_data_yahoo(ticker, startdate, enddate)
            __write_to_file(ticker, __raw_data)

        except Exception as exp:

            print('Error load tickers: ', ticker, 'Exeption: ', exp)
            err_arrsy.append(ticker)

# Write data to output csv file
def __write_to_file(ticker, data):

    __fileDir = os.path.dirname(os.path.abspath(__file__))
    __filename = __fileDir + '/data/stocks/train_' + str(ticker).upper() + '.csv'

    if os.path.isfile(__filename):
        os.remove(__filename)

    with open(__filename, 'a', newline = '') as f:
        data.to_csv(f)
    f.close()


print ("Harvester started ....")

data = D.DataMiner(3)

# 1- long list
# 2 - short list
for __ticker in data.get_tickers(2):

    loadfile('en', __ticker, "", "2000-01-01", "2019-12-31")
    #input("Press any key ...")

if (len(err_arrsy)>0):
    print (str(len(err_arrsy)) + "- tickets wasn't received : ")
    print("----> \n")
    print(err_arrsy)
