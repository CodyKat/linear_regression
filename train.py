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
    return data


# Adjusted the learning rate for example
def linear_regression(learn_rate=1e-3):
    delta = np.array([0.0, 0.0])

    pos = parse_data()
    X, Y = zip(*pos)

    X = np.array(X)
    Y = np.array(Y)
    print("before X: ", X)
    print("before Y: ", Y)

    min_X = np.min(X)
    min_Y = np.min(Y)
    max_X = np.max(X)
    max_Y = np.max(Y)

    X = (X - min_X) / (max_X - min_X)
    Y = (Y - min_Y) / (max_Y - min_Y)
    print("after X: ", X)
    print("after Y: ", Y)

    for i in range(100000):
        # while 1:
        Y_pred = delta[1] * X + delta[0]

        grad = np.array([
            (Y_pred - Y).mean(),
            ((Y_pred - Y) * X).mean()
        ])
        print(grad)

        delta -= learn_rate * grad

        if np.linalg.norm(grad) < 1e-10:
            break

    delta[1] = delta[1] * (max_Y - min_Y) / (max_X - min_X)
    delta[0] = delta[0] * (max_Y - min_Y) + min_Y - delta[1] * min_X
    print(delta)
    plt.scatter(X, Y)
    plt.scatter(X, Y_pred)
    plt.show()
    return delta
