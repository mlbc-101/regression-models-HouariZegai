""" Please inlude the loading, preprocessing and the the regression model. """

# import libraries
import numpy as np
import pandas as pd

# load dataset
data = pd.read_csv('A:/Learn/Projects/ML Projects/MLBC-WithYounes/Practise/2 - regression-models-HouariZegai/boston_housing/housing.csv')
data.head()

# overview data
data.info()
data.describe()

# check if the null exists
data.isnull().sum()

y = data['MEDV']
X = data.drop(['MEDV'], axis = 1)

# preprocessing task

# standarise the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split data (train/test)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_true = train_test_split(X, y, test_size = 0.3, random_state = 11) 

from sklearn.metrics import r2_score
def evaluate(reg):
    y_pred = reg.predict(x_test)    
    print('score of {} are: Accuracy = {}'.format(type(reg).__name__, r2_score(y_true, y_pred)))

# train task
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
evaluate(lr) # 0.6396268709335121

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(x_train, y_train)
evaluate(knr) # 0.7597179673507334

from sklearn.svm import SVR
svm_svr = SVR()
svm_svr.fit(x_train, y_train)
evaluate(svm_svr) # -0.012758120956391172

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()
rf_reg.fit(x_train, y_train)
evaluate(rf_reg) # 0.7489899034319791

from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(x_train, y_train)
evaluate(dt_reg) # 0.6092293932523827