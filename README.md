House Price Prediction
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import seaborn as sns
 df_house = pd.read_csv('Housing Prices.csv')
​
df_house.head()
price	bedrooms	bathrooms	sqft_living	sqft_lot	floors	waterfront	view	condition	grade	sqft_above	sqft_basement	yr_built	yr_renov
0	221900.0	3	1.00	1180	5650	1.0	0	0	3	7	1180	0	1955	0
1	538000.0	3	2.25	2570	7242	2.0	0	0	3	7	2170	400	1951	1991
2	180000.0	2	1.00	770	10000	1.0	0	0	3	6	770	0	1933	0
3	604000.0	4	3.00	1960	5000	1.0	0	0	5	7	1050	910	1965	0
4	510000.0	3	2.00	1680	8080	1.0	0	0	3	8	1680	0	1987	0
df_house.tail()
price	bedrooms	bathrooms	sqft_living	sqft_lot	floors	waterfront	view	condition	grade	sqft_above	sqft_basement	yr_built	yr_renov
21608	360000.0	3	2.50	1530	1131	3.0	0	0	3	8	1530	0	2009	0
21609	400000.0	4	2.50	2310	5813	2.0	0	0	3	8	2310	0	2014	0
21610	402101.0	2	0.75	1020	1350	2.0	0	0	3	7	1020	0	2009	0
21611	400000.0	3	2.50	1600	2388	2.0	0	0	3	8	1600	0	2004	0
21612	325000.0	2	0.75	1020	1076	2.0	0	0	3	7	1020	0	2008	0
df_house.isnull().sum()
price            0
bedrooms         0
bathrooms        0
sqft_living      0
sqft_lot         0
floors           0
waterfront       0
view             0
condition        0
grade            0
sqft_above       0
sqft_basement    0
yr_built         0
yr_renov         0
dtype: int64
sns.pairplot(df_house)
C:\Users\kharw\anaconda3\Lib\site-packages\seaborn\axisgrid.py:118: UserWarning: The figure layout has changed to tight
  self._figure.tight_layout(*args, **kwargs)
<seaborn.axisgrid.PairGrid at 0x2a3612625d0>

df_house.corr()
price	bedrooms	bathrooms	sqft_living	sqft_lot	floors	waterfront	view	condition	grade	sqft_above	sqft_basement	yr_built	yr_renov
price	1.000000	0.308338	0.525134	0.702044	0.089655	0.256786	0.266331	0.397346	0.036392	0.667463	0.605566	0.323837	0.053982	0.126442
bedrooms	0.308338	1.000000	0.515884	0.576671	0.031703	0.175429	-0.006582	0.079532	0.028472	0.356967	0.477600	0.303093	0.154178	0.018841
bathrooms	0.525134	0.515884	1.000000	0.754665	0.087740	0.500653	0.063744	0.187737	-0.124982	0.664983	0.685342	0.283770	0.506019	0.050739
sqft_living	0.702044	0.576671	0.754665	1.000000	0.172826	0.353949	0.103818	0.284611	-0.058753	0.762704	0.876597	0.435043	0.318049	0.055363
sqft_lot	0.089655	0.031703	0.087740	0.172826	1.000000	-0.005201	0.021604	0.074710	-0.008958	0.113621	0.183512	0.015286	0.053080	0.007644
floors	0.256786	0.175429	0.500653	0.353949	-0.005201	1.000000	0.023698	0.029444	-0.263768	0.458183	0.523885	-0.245705	0.489319	0.006338
waterfront	0.266331	-0.006582	0.063744	0.103818	0.021604	0.023698	1.000000	0.401857	0.016653	0.082775	0.072075	0.080588	-0.026161	0.092885
view	0.397346	0.079532	0.187737	0.284611	0.074710	0.029444	0.401857	1.000000	0.045990	0.251321	0.167649	0.276947	-0.053440	0.103917
condition	0.036392	0.028472	-0.124982	-0.058753	-0.008958	-0.263768	0.016653	0.045990	1.000000	-0.144674	-0.158214	0.174105	-0.361417	-0.060618
grade	0.667463	0.356967	0.664983	0.762704	0.113621	0.458183	0.082775	0.251321	-0.144674	1.000000	0.755923	0.168392	0.446963	0.014414
sqft_above	0.605566	0.477600	0.685342	0.876597	0.183512	0.523885	0.072075	0.167649	-0.158214	0.755923	1.000000	-0.051943	0.423898	0.023285
sqft_basement	0.323837	0.303093	0.283770	0.435043	0.015286	-0.245705	0.080588	0.276947	0.174105	0.168392	-0.051943	1.000000	-0.133124	0.071323
yr_built	0.053982	0.154178	0.506019	0.318049	0.053080	0.489319	-0.026161	-0.053440	-0.361417	0.446963	0.423898	-0.133124	1.000000	-0.224874
yr_renov	0.126442	0.018841	0.050739	0.055363	0.007644	0.006338	0.092885	0.103917	-0.060618	0.014414	0.023285	0.071323	-0.224874	1.000000
df_house.head()
price	bedrooms	bathrooms	sqft_living	sqft_lot	floors	waterfront	view	condition	grade	sqft_above	sqft_basement	yr_built	yr_renov
0	221900.0	3	1.00	1180	5650	1.0	0	0	3	7	1180	0	1955	0
1	538000.0	3	2.25	2570	7242	2.0	0	0	3	7	2170	400	1951	1991
2	180000.0	2	1.00	770	10000	1.0	0	0	3	6	770	0	1933	0
3	604000.0	4	3.00	1960	5000	1.0	0	0	5	7	1050	910	1965	0
4	510000.0	3	2.00	1680	8080	1.0	0	0	3	8	1680	0	1987	0
 x = df_house.drop(['price'], axis=1) 
​
 y = df_house['price']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
LinearRegression()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
lr.fit(x_train, y_train)
LinearRegression()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
c = lr.intercept_
m = lr.coef_
​
c
6116114.801079137
m
array([-3.38347201e+04,  4.07307047e+04,  1.11348032e+02, -1.93676429e-01,
        2.97938331e+04,  5.82094300e+05,  4.66535950e+04,  1.86103156e+04,
        1.25810646e+05,  5.07551848e+01,  6.05928473e+01, -3.53718662e+03,
        3.95470340e+00])
y_pred = lr.predict(x_test)
​
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
Mean Squared Error: 48997075702.10165
R-squared: 0.6435567047124281
 x_new = pd.DataFrame({
    'bedrooms': [3],  
    'bathrooms': [2],
    'sqft_living': [1680],
    'sqft_lot': [8080],
    'floors': [1],
    'waterfront': [0],
    'view':[0],
    'condition':[3],
    'grade':[8],
    'sqft_above':[1680],
    'sqft_basement':[0],
    'yr_built':[1987],
    'yr_renov':[0],
 })  
y_new_pred = lr.predict(x_new)
print("Predicted price for new data:", y_new_pred[0])
Predicted price for new data: 430560.6742792595
​
