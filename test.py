import sys
import numpy as np
import dataMaker as d

data = d.DataMaker()

inputDir = data.get_file_dir()+'/data/'

print(" -----> "+ inputDir + str(sys.argv[1]))
array = np.load(inputDir + str(sys.argv[1]))
print (" <<<< "+array.shape[0])
for i in range(array.shape[0]):
    input(array[i])
