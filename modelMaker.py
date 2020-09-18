#!/usr/bin/env python3

import csv
import os
import re
from decimal import Decimal as D, ROUND_DOWN
from datetime import datetime, date
import numpy as np
from shutil import copyfile


class ModelMaker:

    __fileDir = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, testPrefix) -> None:
        super().__init__()
        self.testPrefix = testPrefix

    def get_check_data(self, type: str, caseName: str, shape='3D') -> object:
        """
        :return: X, y numpy arrays
        :rtype: object
        """
        list = ['X', 'y']
        result = []
        data_path = self.__fileDir + '/data/test/cases/binary/'
        for i in list:
            input(i)
            with open(data_path + type + '_'+i+'_' + caseName + '_' + self.testPrefix + '.npy', 'rb') as fin:
                result.append(np.load(fin))

        if(shape == '2D'):
            result[0] = result[0].reshape(result[0].shape[0], -1)

        print(' >>>> Loaded ' + type + ' data case '+ caseName + ' shape X/y :' + str(result[0].shape) + ' ' + str(result[1].shape) )
        return result[0], result[1]

    def get_file_dir(self):
        return self.__fileDir

    def save_conf(self, model, prefix):
        """
        :param model:
        :return: Null
        """
        json_file = open(self.__fileDir + "/data/model_test/weights_"+prefix+".json", "w")
        json_file.write(model.to_json())
        json_file.close

    # Нахождение индекса максимального элемента
    # Если два максимума, тогда возвращается 0
    def __get_max_index(self, vector):

        convert_dict = {0: 1, 1: 0, 2: -1}
        winner = np.argwhere(vector == np.amax(vector))
        if (winner.size > 1):
            return 0
        else:
            return convert_dict[winner[0][0]]

    def __sum_check_results(self, vector):
        up = 0
        none = 0
        down = 0
        for i in range(vector.shape[0]):
            max_index = self.__get_max_index(vector[i])
            if (max_index == 0):
                none += 1
            elif (max_index == 1):
                up += 1
            elif (max_index == -1):
                down += 1
        print("\nUP:\t"+str(up)+"\nNONE:\t"+str(none)+"\nDOWN:\t"+str(down)+"\n")
        return up, none, down

    # Check single model
    def check_single_model(self, y_UP, y_NONE, y_DOWN, model):

        all_errors = 0
        print ("------------------------------------ \n")
        print(">>> Check Up case (shape " + str(y_UP.shape) + "): ")
        up_, none, down = self.__sum_check_results(y_UP)
        all_errors += abs(down)

        print(">>> Check None case shape("+str(y_NONE.shape)+") : ")
        up, none, down = self.__sum_check_results(y_NONE)
        all_errors = all_errors + abs(down) + up

        print(">>> Check Down case shape("+str(y_DOWN.shape)+") : ")
        up, none, down = self.__sum_check_results(y_DOWN)
        all_errors = all_errors + up

        try:
            k1 = 1 - all_errors/(y_UP.shape[0]+y_NONE.shape[0]+y_DOWN.shape[0])
        except ZeroDivisionError:
            k1 = 1000
        try:
            k2 = 1 - all_errors / (up_ + abs(down))
        except ZeroDivisionError:
            k2 = 1000

        print (">>>> Gold : "+str(up_+abs(down))+"\t Shit : " + str(all_errors)+
               "\t Absolute_Error : "+str(k1)+"\t Relevant_Error : "+ str(k2)+"\n")

        self.__archive_model_data(up_+abs(down), all_errors, k1,k2,model)

    def __archive_model_data(self, gold, shit, absErr, relErr, prefix):

        dateTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        outputDir = self.__fileDir + "/data/model_test/archive/"
        filename = outputDir + 'archive_DB.csv'
        model_name = "weights_"+ prefix

        if os.path.exists(filename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        with open(filename, append_write, newline='') as csv_out_file:
            output = csv.writer(csv_out_file, delimiter=';')
            if (append_write == 'w'):
                output.writerow(['Date', 'Gold', 'Shit', 'Rel Error', 'Abs Error', 'Model', 'Comment'])
            file_count = len([name for name in os.listdir(outputDir + 'models')
                              if os.path.isfile(os.path.join(outputDir + 'models', name))])/2

            output.writerow([dateTime, gold, shit, relErr, absErr, model_name + "_"+str(file_count)+".h5"])
        csv_out_file.close()

        copyfile(self.__fileDir+ "/data/model_test/"+model_name+".json",
                 outputDir+"models/"+model_name + "_"+str(file_count)+".json" )

        copyfile(self.__fileDir + "/data/model_test/" + model_name + ".h5",
                 outputDir + "models/" + model_name + "_" + str(file_count) + ".h5")
