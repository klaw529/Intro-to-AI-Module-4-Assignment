import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

data = pd.read_csv('Real estate.csv')

x = data.loc[:,['X1 transaction date','X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores','X5 latitude','X6 longitude']]
y = data.loc[:, 'Y house price of unit area']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_predict=  regressor.predict(X_test)

percent_error = ((np.sum(Y_test) - np.sum(y_predict))/np.sum(Y_test))*100
print(percent_error)
MSE = metrics.mean_squared_error(Y_test, y_predict)
print(MSE)
