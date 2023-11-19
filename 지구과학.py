#필요한 라이브러리 불러오기
from sklearn.linear_model import LinearRegression as lr
import pandas as pd
import matplotlib.pyplot as plt

#csv 형식의 데이터 불러오기
A=pd.read_csv("C:/Users/82108/OneDrive/바탕   화면/울산.csv",encoding='cp949')

#X축과 Y축 정하기
X=A[['날짜']]
Y=A[['실측값']]

#불러온 데이터로 선형 회귀 분석
line_fitter=lr()
line_fitter.fit(X,Y)
predicted=line_fitter.predict(X)

#날짜별 실측값 그래프와 회귀 분석을 통한 추세선 표현
plt.plot(X,Y)
plt.plot(X,predicted)
plt.show()

#회귀분석된 추세선을 바탕으로 100년 뒤에 해수면 예측
print(f'{round(line_fitter.coef_[0][0]*5200+line_fitter.intercept_[0],3)}cm')
