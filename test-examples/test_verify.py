#!/usr/bin/env python3.6
import datama_test as d

# Загрузка проверочных данных
data = d.DataManager("test", 5, 1)
X_test, y_test = data.get_data_(0, 11)

print(X_test)
print('----------------------')
print(y_test)
print('==================================================')

X_test, y_test = data.get_data(0, 11)

print(X_test)
print('----------------------')
print(y_test)
