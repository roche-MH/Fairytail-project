# prophet



prophet 은 페이스북에서 개발한 시계열 예측 패키지이다.

ARIMA와 같은 확률론적이고 이론적인 모형이 아니라 몇가치 경험적 규칙을 사용하는 단순 회귀모형이지만 단기적 예측에서는 큰 문제 없이 사용할 수 있다.



## 설치

```python
conda install pystan
conda install -c conda-forge fbprophet
```



## 사용법

```python
import pandas as pd

url = "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_peyton_manning.csv"
df = pd.read_csv(url)
df.tail()
```

|      |         ds |         y |
| :--- | ---------: | --------: |
| 2900 | 2016-01-16 |  7.817223 |
| 2901 | 2016-01-17 |  9.273878 |
| 2902 | 2016-01-18 | 10.333775 |
| 2903 | 2016-01-19 |  9.125871 |
| 2904 | 2016-01-20 |  8.891374 |



```python
# prophet의 로그 기능 끄기
import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)
```

사용법은 간단하다. Prophet 클래스 객체를 만들고 시계열 데이터를 입력으로 fit 메서드를 호출한다.



```python
from fbprophet import Prophet

m = Prophet()
m.fit(df)
```

```
ERROR:fbprophet:Importing plotly failed. Interactive plots will not work.
<fbprophet.forecaster.prophet at 0x7f3b30092f28>
```



예측을 하려면 다음 메서드를 사용한다.

- `make_future_dataframe`: 예측 날짜 구간 생성
- `predict`: 신뢰 구간을 포함한 예측 실행

```python
future = m.make_future_dataframe(periods=365) #예측하고자 하는 날수
future.tail()
```

| ds   |            |
| :--- | ---------- |
| 3265 | 2017-01-15 |
| 3266 | 2017-01-16 |
| 3267 | 2017-01-17 |
| 3268 | 2017-01-18 |
| 3269 | 2017-01-19 |

```python
forecast = m.predict(future)
forecast.tail()
```

| ds   |      trend | yhat_lower | yhat_upper | trend_lower | trend_upper | additive_terms | additive_terms_lower | additive_terms_upper |   weekly | weekly_lower | weekly_upper |    yearly | yearly_lower | yearly_upper | multiplicative_terms | multiplicative_terms_lower | multiplicative_terms_upper | yhat |          |
| :--- | ---------: | ---------: | ---------: | ----------: | ----------: | -------------: | -------------------: | -------------------: | -------: | -----------: | -----------: | --------: | -----------: | -----------: | -------------------: | -------------------------: | -------------------------: | ---: | -------- |
| 3265 | 2017-01-15 |   7.180847 |   7.435809 |    8.948534 |    6.827810 |       7.552677 |             1.018428 |             1.018428 | 1.018428 |     0.048295 |     0.048295 |  0.048295 |     0.970133 |     0.970133 |             0.970133 |                        0.0 |                        0.0 |  0.0 | 8.199274 |
| 3266 | 2017-01-16 |   7.179809 |   7.755114 |    9.291769 |    6.825133 |       7.552981 |             1.344435 |             1.344435 | 1.344435 |     0.352287 |     0.352287 |  0.352287 |     0.992148 |     0.992148 |             0.992148 |                        0.0 |                        0.0 |  0.0 | 8.524244 |
| 3267 | 2017-01-17 |   7.178771 |   7.530683 |    9.036235 |    6.822456 |       7.553219 |             1.132844 |             1.132844 | 1.132844 |     0.119624 |     0.119624 |  0.119624 |     1.013220 |     1.013220 |             1.013220 |                        0.0 |                        0.0 |  0.0 | 8.311615 |
| 3268 | 2017-01-18 |   7.177733 |   7.413488 |    8.845561 |    6.820041 |       7.553632 |             0.966499 |             0.966499 | 0.966499 |    -0.066647 |    -0.066647 | -0.066647 |     1.033146 |     1.033146 |             1.033146 |                        0.0 |                        0.0 |  0.0 | 8.144232 |
| 3269 | 2017-01-19 |   7.176695 |   7.425690 |    8.894338 |    6.818071 |       7.553324 |             0.979396 |             0.979396 | 0.979396 |    -0.072284 |    -0.072284 | -0.072284 |     1.051680 |     1.051680 |             1.051680 |                        0.0 |                        0.0 |  0.0 | 8.156091 |

```python
forecast.iloc[-365:, :].yhat.plot()
```

![image-20200224214122528](C:\Users\s_m04\OneDrive\바탕 화면\kaggle\코로나 예측\image-20200224214122528.png)

다음 메서드를 사용하면 시계열을 시각화할 수 있다.

- `plot`: 원래의 시계열 데이터와 예측 데이터
- `plot_components`: 선형회귀 및 계절성 성분별로 분리



```python
fig1 = m.plot(forecast, uncertainty=False)
plt.show()
fig2 = m.plot_components(forecast)
plt.show()
```

![image-20200224214245870](C:\Users\s_m04\OneDrive\바탕 화면\kaggle\코로나 예측\image-20200224214245870.png)

![image-20200224214308410](C:\Users\s_m04\OneDrive\바탕 화면\kaggle\코로나 예측\image-20200224214308410.png)



### 기본 원리[¶](https://datascienceschool.net/view-notebook/8903aa20770746e78fb5b1834ab5334b/#기본-원리)

Prophet은 다음 순서로 시계열에 대한 회귀분석 모형을 만든다.

- 시간 데이터의 각종 특징을 임베딩해서 계절성 추정을 한다.
- 나머지 데이터는 구간별 선형회귀(piecewise linear regression) 분석을 한다.

선형 회귀분석은 전체 시계열의 앞 80%부분을 25개의 구간으로 나누어 실시한다. 구간 구분점(change point)는 `changepoints` 속성에 있다.

[페이스북github](https://facebook.github.io/prophet/docs/trend_changepoints.html)