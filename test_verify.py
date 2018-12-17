#!/usr/bin/env python3.6
import datama_new as d

# Загрузка проверочных данных
data = d.DataManager("USDRUB-w", 5, 1)
X_test, y_test = data.get_edu_data()

print(X_test)
print('-------------------------')
print(y_test)
print("=========================")
print(data.denorm_y_array(y_test))
print("=========================")
y_test_reshaped = data.reshapy_y_by_coll(y_test, 1)
print("--Reshape--")
print(y_test_reshaped)
print(data.denorm_y(y_test_reshaped))
