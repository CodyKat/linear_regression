주행거리(km)로 부터 차의 가격(price)를 예측하는 선형회귀 프로그램입니다.

HOW TO DO
1. km, price 가 있는 data 파일을 준비합니다.
2. python3 train.py 명령어를 통해 모델 학습 프로그램을 실행합니다.
   - data 파일의 파일명을 입력합니다.(default == data.csv)
   - 실시간으로 선형회귀 학습을 하는 시뮬레이터가 실행됩니다.
   - 결과물로 result.txt 라는 파일이 생성됩니다.
3. python3 predict.py 명령어를 통해 예측 프로그램을 실행합니다.
   - 주행거리(km)를 입력합니다.
   - result.txt 파일을 읽고 예측되는 차의 가격을 출력합니다.
