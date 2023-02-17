# **0. 대회 소개**
- 보스턴 집값 예측 경진대회
- 보스턴의 집값을 예측
- 여러 회귀 모형 활용
- 모델 평가 기준 학습

# **1. 데이터 준비**
- CRIM: 도시별 1인당 범죄율
- ZN: 25,000 피트를 초과하는 주거용 토지의 비율
- NDUS: 비상업 면적의 비율
- CHAS: 찰스강에 대한 더미 변수(강의 경계에 위치한 경우는 1, 아니면 0)
- NOX: 일산화질소 농도
- RM: 주택당 평균 방의 개수
- AGE: 1940년 이전에 건축된 자가주택의 비율
- DIS: 5개의 보스턴 고용 센터와의 거리
- RAD: 고속도로 접근성 지수
- TAX: 10,000달러당 재산세율
- PTRATIO: 도시별 교사와 학생 수 비율
- B: 마을의 흑인 거주 비율
- LSTAT: 하위 계층의 비율
- MEDV: 본인 소유의 주택 가격(중앙값, 단위: 천달러)

# **2. 데이터 전처리**
- ```data.describe()```로 통계치 확인 결과 데이터 간의 **스케일 차이**가 존재함을 확인할 수 있었음
- ```data.hist()```로 데이터 분포 확인 결과 데이터가 **왜곡**된 분포를 가지고 있음을 확인할 수 있었음

- 선형 회귀 모델과 같은 선형 모델의 경우 일반적으로 피처와 타깃값 간에 선형의 관계가 있다고 가정하고, 최적의 선형함수를 찾아내 결과값을 예측
  - feature와 target 데이터의 분포가 **정규 분포** 형태인 것을 선호
> 다양한 데이터 전처리 시도

### **2-1. 표준화(StandardScaler)**
- 분석 시 변수들의 스케일이 다른 경우 컬럼 별 단위 또는 범위를 통일시켜주기 위해 표준화를 수행
- **표준화**: 데이터 값들을 평균이 0이고 분산이 1인 표준 정규 분포로 만드는 것

### **2-2. 정규화(MinMaxScaler)**
- 최솟값이 0이고 최댓값이 1인 값으로 정규화

### **2-3. 로그 변환(Log Transformation)**
- 원본 값에 **로그 함수**(```log1p()```)를 적용해 보다 정규분포에 가까운 형태로 값의 분포를 변경
- 타겟 데이터(target data)의 경우 주로 로그 변환 수행
  - 예측 후 원래 값으로 되돌리기 위해 ```np.expm1()``` 함수 활용 

# **3. 평가 지표**
- 회귀 모델의 성능을 평가하기 위해 여러 평가 지표들을 활용할 수 있음

### **3-1. MSE(Mean Squared Error)**
- 예측값과 실제값의 차이의 제곱에 대하여 평균을 낸 값
- $\frac{1}{n} \sum_{i = 1}^{n} (y_{i} - x_{i})^{2}$
- 코드
```Python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(pred,actual)
```

### **3-2. RMSE(Root Mean Squared Error)**
- 예측값과 실제값의 차이에 대한 제곱에 대하여 평균을 낸 후 루트를 씌운 값
- $\sqrt{(\frac{1}{n})\sum_{i=1}^{n}(y_{i} - x_{i})^{2}}$
- 코드
```Python
import numpy as np
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(pred,actual))
```

# **4. 회귀분석(Regression)**
- 독립변수(x)로 종속변수(y)를 예측하는 것
  - 독립변수: 변수의 변화 원인이 모형 밖에 있는 변수
  - 종속변수: 변수의 변화 원인이 모형 내에 있는 변수

### **4-1. 선형 회귀(LinearRegression)**
- 실제값과 예측값의 RSS(Residual Sum of Squares)를 최소화 해 OLS(Ordinary Least Squares) 추정 방식으로 구현
- 규제를 적용하지 **않은** 모델
- 코드
```Python
from sklearn.linear_model import linearRegression

model = LinearRegression(n_jobs = -1) # CPU Core를 있는 대로 모두 사용하겠다.
model.fit(X_train, y_train) # 학습
pred = model.predict(X_test) # 예측
mean_squared_error(pred, y_test) # 평가
```

### **4-2. 규제(Regularization)**
- 학습이 과적합되는 것을 방지하고자 일종의 penalty를 부여하는 것
#### **1) L1 규제**  
- 가중치의 합을 더한 값에 규제 강도를 곱하여 오차에 더한 값($Error=MSE+α|w|$)
- 어떤 가중치는 실제로 0이 됨 -> 모델에서 완전히 제외되는 특성이 발생할 수 있음
- **라쏘(Lasso)** 모델에 적용됨
- 코드
```Python
from sklearn.linear_model import Lasso

model = Lasso(alpha = alpha) # alpha: 규제 강도
model.fit(X_train, y_train) # 학습
pred = model.predict(X_test) # 예측
mean_squared_error(pred, y_test) # 평가
```

#### **2) L2 규제**  
- 각 가중치 제곱의 합에 규제 강도를 곱한 값($Error=MSE+αw^2$)
- 규제 강도를 크게 하면 가중치가 더 많이 감소되고(규제를 중요시함), 규제 강도를 작게 하면 가중치가 증가함(규제를 중요시하지 않음)
- **릿지(Ridge)** 모델에 적용됨
- 코드
```Python
from sklearn.linear_model import Ridge

model = Ridge(alpha = alpha) # 규제 강도
model.fit(X_train, y_train) # 학습
pred = model.predict(X_test) # 예측
mean_squared_error(pred, y_test) # 평가
```

#### **3) 엘라스틱넷(ElasticNet)**
- L1 규제 + L2 규제
- l1_ratio(default: 0.5) 속성 -> 규제 강도 조정
  - l1_ratio = 0: L2 규제만
  - l1_ratio =1: L1 규제만
  - 0 < l1_ratio < 1: L1 and L2 규제(혼합 사용)
- 코드
```Python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha = alpha,l1_ratio = l1_ratio) # 규제 강도
model.fit(X_train, y_train) # 학습
pred = model.predict(X_test) # 예측
mean_squared_error(pred, y_test) # 평가
```

### **4-3. 다항 회귀**
- 다항식의 계수 간 상호작용을 통해 새로운 feature를 생성
- 데이터가 단순한 직선의 형태가 아닌 비선형 형태여도 선형 모델을 사용하여 비선형 데이터를 학습할 수 있음
  - 원본 데이터: 선형 관계가 x
  - 새로운 feature: 선형 관계
- 특성의 거듭제곱을 새로운 특성으로 추가하고 확장된 특성을 포함한 데이터 셋에 선형 모델을 학습
- 코드
```Python
from sklearn.preprocessing import PolynomialFeatures

poly = PolyNomialFeatures(degree = 2, include_bias = False)
# degree: 차수(몇 제곱까지 갈 것인가)
# include_bias: 절편 포함 여부 선택
```

#### **파이프라인(PipeLine)**
- 여러 가지 방법들을 융합하는 기법
- 코드
```Python
from sklearn.pipeline import make_pipeline

# 파이프라인 생성(모델 객체 생성)
pipeline = make_pipeline(
  StandarsScaler(),
  ElasticNet(alpha = 0.1, l1_ratio = 0.2)
)

pipeline_pred = pipeline.fit(X_train, y_train).predict(X_test) # 학습, 예측
mean_squared_error(pipeline_pred,y_test) # 평가
```

# **5. 결과 정리**
- 4가지 버전으로 실험 수행  

|   |**ver1**|**ver2**|**ver3**|**ver4**|
|----------|-------|-------|-------|-------|
|**feature 전처리**|표준화|정규화|표준화|정규화|
|**target 로그 변환**|X|X|O|O|

- 각 과정마다 최적의 회귀모형이 다른 것을 확인할 수 있었음
  - 4가지 버전 중 최적 회귀모형: ver3의 Polynomial Lasso(alpha = 100)
  -  
