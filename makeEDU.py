#!/usr/bin/env python3
import dataMiner as d
import sys


if (len(sys.argv) < 2):
    print("Argument not found ")
    exit(0)

data = d.DataMiner(3)
# 1- long list
# 2 - short list
data.make_edu_data(2, '_last_'+sys.argv[1])


