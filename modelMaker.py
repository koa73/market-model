#!/usr/bin/env python3

import csv
import os
from datetime import datetime
import numpy as np
from shutil import copyfile
import tensorflow as tf
from ConcatLayer_3 import ConcatLayer
from keras.utils import CustomObjectScope


class ModelMaker:

    __fileDir = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, testPrefix='') -> None:
        super().__init__()
        self.testPrefix = testPrefix

    def model_loader(self, prefix, path=''):
        if not path:
            input_dir = self.__fileDir + '/data/model_test/archive/models/'
        else:
            input_dir = self.__fileDir + path
        print("\n >>>>>>> Load file : " + input_dir + prefix + ".json  .....\n")
        json_file = open(input_dir + prefix + ".json", "r")
        model_json = json_file.read()
        json_file.close()
        model = tf.keras.models.model_from_json(model_json,  custom_objects={'ConcatLayer': ConcatLayer})
        print("\n >>>>>>> Load file : " + input_dir + prefix + ".h5  .....\n")
        model.load_weights(input_dir + prefix +".h5",  custom_objects={'ConcatLayer': ConcatLayer})
        model.trainable = False
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def get_check_data(self, type: str, caseName: str, shape='3D'):

        data_path = self.__fileDir + '/data/test/cases/binary/'
        return self.__get_data(type, caseName, data_path, shape)

    def get_edu_data(self, type: str, caseName: str, shape='3D'):
        data_path = self.__fileDir + '/data/'
        return self.__get_data(type, caseName, data_path, shape)


    def __get_data(self, type: str, caseName: str, data_path: str, shape) -> object:
        """
        :return: X, y numpy arrays
        :rtype: object
        """
        _list = ['X', 'y']
        result = []

        for i in _list:
            with open(data_path + type + '_' + i + '_' + caseName + '.npy', 'rb') as fin:
                result.append(np.load(fin))

        if(shape == '2D'):
            result[0] = result[0].reshape(result[0].shape[0], -1)

        print(' >>>> Loaded ' + type + ' data case '+ caseName + ' shape X/y :' + str(result[0].shape) + ' '
              + str(result[1].shape) )
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
    def check_single_model(self, y_UP, y_NONE, y_DOWN, model, comment=''):

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

        self.__archive_model_data(up_+abs(down), all_errors, k1, k2, model, comment)

    def __archive_model_data(self, gold, shit, absErr, relErr, prefix, comment):

        dateTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        outputDir = self.__fileDir + "/data/model_test/archive/"
        filename = outputDir + 'archive_DB.csv'
        model_name = "weights_"+ prefix

        if os.path.exists(filename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        try:
            file_count = len([name for name in os.listdir(outputDir + 'models')
                              if os.path.isfile(os.path.join(outputDir + 'models', name))]) / 2

            copyfile(self.__fileDir+ "/data/model_test/"+model_name+".json",
                 outputDir+"models/"+model_name + "_"+str(int(file_count))+".json" )

            copyfile(self.__fileDir + "/data/model_test/" + model_name + ".h5",
                 outputDir + "models/" + model_name + "_" + str(int(file_count)) + ".h5")

            with open(filename, append_write, newline='') as csv_out_file:
                output = csv.writer(csv_out_file, delimiter=';')
                if (append_write == 'w'):
                    output.writerow(['Date', 'Gold', 'Shit', 'Rel Error', 'Abs Error', 'Model', 'Comment'])

                output.writerow(
                    [dateTime, gold, shit, relErr, absErr, model_name + "_" + str(int(file_count)) + ".h5", comment])
            csv_out_file.close()

        except FileNotFoundError:
            print ("Error can't write file")
