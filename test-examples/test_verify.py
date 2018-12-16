#!/usr/bin/env python3.6
import dataman_new as d


# Загрузка проверочных данных
data = d.DataManager("USDRUB-v", 5, 3)
x_train, y_train = data.get_edu_data()

print(y_train)
print("----------------------------------------------------------")
print(y_train)
print("==========================================")
print(x_train)
print('*********************************************')
print(data.denorm_x_array(x_train))
