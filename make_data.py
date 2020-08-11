#!/usr/bin/env python3
import dataMaker as d
import sys


if (len(sys.argv) < 3):
    print("Add arguments : type ('edu'/'test'), list_num (0,1,2), prefix, break_size, factor")

    exit(0)

data = d.DataMaker()
#data.get_Xy_arrays('edu', 0, 'b20')
data.get_Xy_arrays(str(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])