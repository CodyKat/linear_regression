from train import linear_regression

def main():
    delta0 = 0
    delta1 = 0
    mileage = int(input())
    linear_regression()

    meta_data = open('result', 'r')

    delta0 = float(meta_data.readline())
    delta1 = float(meta_data.readline())
    print(delta0 + delta1 * mileage)
    

if __name__ == "__main__":
    main()