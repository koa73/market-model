import dataman_new as d


# Загрузка проверочных данных
data = d.DataManager("USDRUB-v", 5, 3)
x_train, y_train = data.get_edu_data(True)

print(y_train)
print("----------------------------------------------------------")
print(x_train)
print("==========================================")
print(data.denorm_x_array(x_train))


