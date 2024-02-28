import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


def parse_data():
    data_size = 0
    data = []
    with open("data.csv", mode='r') as data_file:
        line = data_file.readline()
        while line:
            line = data_file.readline()
            if line == '':
                break
            line = line.split(',')
            line[1] = line[1].strip()
            list_line = [int(i) for i in line]
            data.append(list_line)
            data_size += 1
    data_file.close()
    return data, data_size


# Adjusted the learning rate for example
def linear_regression(learn_rate=1e-10, iteration=10):
    data, data_size = parse_data()
    delta0 = 0
    delta1 = 0
    X, Y = zip(*data)

    X = np.array(X)
    Y = np.array(Y)

    for epoch in range(iteration):
        Y_predict = delta1 * X + delta0

        # error = np.abs(Y_predict - Y).mean()
        # if error < 0.001:
        #     break

        delta0_grad = learn_rate * (Y_predict - Y).mean()
        delta1_grad = learn_rate * ((Y_predict - Y) * X).mean()
        print(delta0_grad, delta1_grad)

        delta0 = delta0 - delta0_grad
        delta1 = delta1 - delta1_grad

    with open("result", 'w') as file:
        file.write(str(delta0))
        file.write("\n")
        file.write(str(delta1))

    plt.scatter(X, Y)
    plt.scatter(X, Y_predict)
    plt.show()

    return delta0, delta1
