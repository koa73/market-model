#!/usr/bin/env python3
#
import dataMaker as d
import sys

if (len(sys.argv) < 3):
    print("Argument not found. Set arguments : type and source list")
    exit(0)

data = d.DataMaker()
data.prepare_data(sys.argv[1], sys.argv[2])