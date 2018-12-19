#!/usr/bin/env python3.6
import datama as d

# Загрузка проверочных данных
data = d.DataManager("USDRUB", 5, 1)
X_test, y_test = data.get_edu_data()


print(y_test)
print('==================================================')
