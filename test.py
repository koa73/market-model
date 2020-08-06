#!/usr/bin/env python3
#
import dataMaker as d

data = d.DataMaker()
data.prepare_data('edu')
print("--------->")
data.prepare_data('test', 2)