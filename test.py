import sys
import numpy as np
import dataMaker as d

data = d.DataMaker()

input = data.get_file_dir()+'/data/'
array =  np.load(input+str(sys.argv[2]))
for i in range(array.shape[0]):
    input(array[i])
