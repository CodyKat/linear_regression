def predict():
    print("WARNNING : result.txt must be included in current folder")
    try:
        mileage = int(input("please input mileage : "))
    except ValueError:
        print("INPUT YOUR MILEAGE")
        exit()

    try:
        with open('result.txt', 'r') as meta_data:
            delta = meta_data.readline()
            r2 = float(meta_data.readline())
            delta = delta.strip("[]'\n").split()
    except FileNotFoundError:
        delta = [0, 0]
        r2 = 0

    print("delta[0] : ", delta[0])
    print("delta[1] : ", delta[1])

    delta = [float(i) for i in delta]

    print("predict : ", delta[0] + delta[1] * mileage)
    print("r2 : ", r2)

if __name__ == "__main__":
    predict()
