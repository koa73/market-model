#!/usr/bin/env python3
import dataMiner as d

data = d.DataMiner(3)
tikers = data.make_test_case_prepare('test/','test/rawdata/', ['AAPL', 'MSFT'])
data.male_test_data_binary(tikers)