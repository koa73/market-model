#!/usr/bin/env python3
#
import dataMaker as d
import sys

if (len(sys.argv) < 3):
    print("Argument not found. Set arguments : type and source list")
    exit(0)

data = d.DataMaker()
#data.prepare_data('edu', 0)
if (str(sys.argv[1]) == 'custom'):
    data.add_tikers(str(sys.argv[2]))
    data.prepare_data(str(sys.argv[1]), 3)
else:
    data.prepare_data(str(sys.argv[1]), int(sys.argv[2]))
