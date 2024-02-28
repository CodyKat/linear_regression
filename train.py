import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


def parse_data():
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
    data_file.close()
    return data


# Adjusted the learning rate for example
def linear_regression(learn_rate=1e-10):
    data = parse_data()
    delta0 = 0
    delta1 = 0
    prev_delta0_grad = 0
    prev_delta1_grad = 0
    X, Y = zip(*data)

    X = np.array(X)
    Y = np.array(Y)
    prevY = np.array(Y)

    while 1:
        Y_predict = delta1 * X + delta0

        error = np.abs(Y_predict - Y).mean()
        if error < 0.001:
            break

        delta0_grad = learn_rate * (Y_predict - Y).mean()
        delta1_grad = learn_rate * ((Y_predict - Y) * X).mean()

        delta0 = delta0 - delta0_grad
        delta1 = delta1 - delta1_grad

        print(Y_predict - prevY)
        print("\n")
        print(delta0 - prev_delta0_grad)
        learn_rate = (Y_predict - prevY).T @ (delta0 - prev_delta0_grad)

        prev_delta0_grad = delta0_grad
        prev_delta1_grad = delta1_grad
        prevY = Y_predict

    with open("result", 'w') as file:
        file.write(str(delta0))
        file.write("\n")
        file.write(str(delta1))

    plt.scatter(X, Y)
    plt.scatter(X, Y_predict)
    plt.show()

    return delta0, delta1
