import numpy as np

__array = [0.58, 0.14, 0.28, 0.33, 0.15, 0.52, 0.16, 0.06, 0.78]


def concat_result(array):

    convert_dict = {0: 1, 1: 0, 2: -1}

    vector_up = np.array(array).astype(np.float32)[0:3]
    vector_none = np.array(array).astype(np.float32)[3:6]
    vector_down = np.array(array).astype(np.float32)[6:9]

    max_index_up = convert_dict[np.argmax(vector_up, axis=0)]
    max_index_none = convert_dict[np.argmax(vector_none, axis=0)]
    max_index_down = convert_dict[np.argmax(vector_down, axis=0)]

    calc_value = abs(max_index_none) * (max_index_up + max_index_down + max_index_none)

    print (calc_value)

    if (calc_value == 0):
        return np.array([0, 1, 0])
    elif (calc_value >= 1):
        return np.array([1, 0, 0])
    elif (calc_value <= -1):
        return np.array([0, 0, 1])

print(concat_result(__array))