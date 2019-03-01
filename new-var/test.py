import dataprepare as d

data = d.DataPrepare("USDRUB", 5, 1)

X_edu, Y_edu = data.get_edu_data()

for i in range(len(X_edu)):
    print(X_edu[i])
