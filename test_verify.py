#!/usr/bin/env python3.6
import datama_new as d

# Загрузка проверочных данных
data = d.DataManager("USDRUB_TOM_2", 5, 1)
X_test, y_test = data.get_edu_data()

print(data.get_mean())
print("---------------------------")
print(data.get_std())