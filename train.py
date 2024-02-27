import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


learn_rate = 1e-10
data_size = 0


def parse_data():
    global data_size
    data = []
    with open("data.csv", mode='r') as data_file:
        line = data_file.readline()
        while line:
            line = data_file.readline()
            list_line = line.strip().split(',')
            data.append(list_line)
            data_size += 1
    data.pop(-1)
    data_size -= 1
    data_file.close()
    return data


def linear_regression():
    delta0 = 0
    delta1 = 0
    data = parse_data()
    cur_delta0 = 0
    cur_delta1 = 0

    for i in range(10000):
        for line in data:
            y = cur_delta0 + cur_delta1 * int(line[0])
            delta0 -= y - int(line[1])
            delta1 -= (y - int(line[1])) * int(line[0])
        delta0 = delta0 * learn_rate / data_size
        delta1 = delta1 * learn_rate / data_size
        cur_delta0 += delta0
        cur_delta1 += delta1
    outfile = open("result", 'w')
    outfile.write(str(cur_delta0))
    outfile.write("\n")
    outfile.write(str(cur_delta1))
    return cur_delta0, cur_delta1
