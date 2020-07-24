#!/usr/bin/env python3
import dataMiner as d


data = d.DataMiner(3)
tikers = data.make_test_case_prepare('test/','test/rawdata/', 'TSLA')
#print(tikers)
data.make_test_data(tikers, '24.06.2020', 6)