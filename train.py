import numpy as np
import matplotlib.pyplot as plt


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


def linear_regression(learn_rate=1):
    delta = np.array([0.0, 0.0])

    pos = parse_data()
    X, Y = zip(*pos)

    X = np.array(X)
    Y = np.array(Y)

    min_X = np.min(X)
    min_Y = np.min(Y)
    max_X = np.max(X)
    max_Y = np.max(Y)

    X = (X - min_X) / (max_X - min_X)
    Y = (Y - min_Y) / (max_Y - min_Y)

    plt.ion()
    fig, ax = plt.subplots()
    while 1:
        Y_pred = delta[1] * X + delta[0]

        grad = np.array([
            (Y_pred - Y).mean(),
            ((Y_pred - Y) * X).mean()
        ])
        print("gradient X : ", grad[0])
        print("gradient Y : ", grad[1], "\n")

        delta -= learn_rate * grad
        ax.cla()
        ax.plot(X, Y_pred, "red")
        ax.scatter(X, Y)

        plt.show()
        plt.pause(0.0001)
        if np.linalg.norm(grad) < 1e-3:
            break

    delta[1] = delta[1] * (max_Y - min_Y) / (max_X - min_X)
    delta[0] = delta[0] * (max_Y - min_Y) + min_Y - delta[1] * min_X
    Y_min = np.mean(Y)

    nomerator = np.sum(np.square(Y_pred - Y))
    denomerator = np.sum(np.square(Y_pred - Y_min))
    r2 = 1 - nomerator / denomerator

    with open("result.txt", 'w') as data_file:
        data_file.write(str(delta))
        data_file.write("\n")
        data_file.write(str(r2))
    
    print("finish")
    plt.pause(3)
    return delta
