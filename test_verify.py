#!/usr/bin/env python3.6
import datama as d

# Загрузка проверочных данных
data = d.DataManager("USDRUB_TOM_1", 5, 1)
X_test, y_test = data.get_edu_data()
X_v, y_v = data.get_test_data()


print(y_v)
print('==================================================')
print(X_test[-1:])
