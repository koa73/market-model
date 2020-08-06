#!/usr/bin/env python3
import dataMaker as d
import sys


if (len(sys.argv) < 2):
    print("Argument not found.")
    #exit(0)

data = d.DataMaker()
data.get_Xy_arrays('edu', 1, '')