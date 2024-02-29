from train import linear_regression
import numpy as np


def main():
    mileage = int(input())
    # delta = np.array([0.0, 0.0])
    delta = linear_regression()

    # meta_data = open('result', 'r')

    # delta0 = float(meta_data.readline())
    # delta1 = float(meta_data.readline())
    print(delta[0] + delta[1] * mileage)


if __name__ == "__main__":
    main()
