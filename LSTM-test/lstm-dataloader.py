import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import os
import datetime
import lstmdataman

"""
Утилита для загрузки и подготовки данных для индекса
"""

market_identifier = 'TQBR'
start_date = '2010-01-01'
end_date = '2018-12-10'
separator = ','

lstmdataman.loadfile('ALRS', market_identifier, start_date, end_date)
lstmdataman.loadfile('CHMF', market_identifier, start_date, end_date)
lstmdataman.loadfile('FIVE', market_identifier, start_date, end_date)
lstmdataman.loadfile('GAZP', market_identifier, start_date, end_date)
lstmdataman.loadfile('GMKN', market_identifier, start_date, end_date)
lstmdataman.loadfile('LKOH', market_identifier, start_date, end_date)
lstmdataman.loadfile('MGNT', market_identifier, start_date, end_date)
lstmdataman.loadfile('MTSS', market_identifier, start_date, end_date)
lstmdataman.loadfile('NVTK', market_identifier, start_date, end_date)
lstmdataman.loadfile('ROSN', market_identifier, start_date, end_date)
lstmdataman.loadfile('SBER', market_identifier, start_date, end_date)
lstmdataman.loadfile('SNGS', market_identifier, start_date, end_date)
lstmdataman.loadfile('TATN', market_identifier, start_date, end_date)
lstmdataman.loadfile('VTBR', market_identifier, start_date, end_date)
lstmdataman.loadfile('YNDX', market_identifier, start_date, end_date)

