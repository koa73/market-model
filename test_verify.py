#!/usr/bin/env python3.6
import datama as d

# Загрузка проверочных данных
data = d.DataManager("USDRUB", 5, 1)
X_test, y_test = data.get_edu_data()
X_v, y_v = data.get_test_data()


print(data.denorm_y_array(y_v))
print('==================================================')
print(data.denorm_x_array(X_test[-1:]))
