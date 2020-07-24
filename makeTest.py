#!/usr/bin/env python3
import dataMiner as d


data = d.DataMiner(3)
tikers = data.make_test_case_prepare('test/','test/rawdata/', 'MSFT')
#print(tikers)
data.make_test_data(tikers, '31.01.2020', 6)