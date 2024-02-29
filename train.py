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
def linear_regression(learn_rate=1e-3):
    delta = np.array([0.0, 0.0])
    prev_grad = np.array([0.0, 0.0])
    prev_delta = np.array([0.0, 0.0])

    pos = parse_data()
    X, Y = zip(*pos)

    X = np.array(X)
    Y = np.array(Y)

    for i in range(1000):
        Y_pred = delta[1] * X + delta[0]

        grad = np.array([
            (Y_pred - Y).mean(),
            ((Y_pred - Y) * X).mean()
        ])
#       Y_pred는 현재 직선에서 예측한 모든 예측점들의 모임임 그러니까 직선이라고 볼 수 있음

        s = delta - prev_delta  # 파라미터 차이 벡터
        y = grad - prev_grad  # 그라디언트 차이 벡터
        print("s shape : ", s.shape)
        print("y shape : ", y.shape)
        print("np.dot(s, y) : ", np.dot(s, y))
        print(np.linalg.norm(y))

        if i != 0:
            learn_rate = np.abs(np.dot(s, y) / np.linalg.norm(y)**2)
        # 이전 그라디언트 업데이트
        prev_delta = np.copy(delta)
        prev_grad = np.copy(grad)
        delta -= learn_rate * grad
        print("learn_rate : ", learn_rate)
        if learn_rate < 1e-10:
            break
    print(delta)
    return delta
