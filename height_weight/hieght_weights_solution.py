""" Please inlude the loading, preprocessing and the the regression model. """

# import libraries
import numpy as np
import pandas as pd

# load dataset
data = pd.read_csv('A:/Learn/Projects/ML Projects/MLBC-WithYounes/Practise/2 - regression-models-HouariZegai/height_weight/height_weight_small.csv')
data.head()

# check missing data (if exist)
data.isnull().sum()

# overview data
for i in data.columns:
    print('{} : {}'.format(i, data[i].unique()))

data.shape

data = data.drop(['Index'], axis = 1)
y = data.drop(['height'], axis = 1)
X = data.drop(['weights'], axis = 1)

# preprocessing task
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_true = train_test_split(X, y, test_size = 0.3, random_state = 16)

# evaluate the model
from sklearn.metrics import r2_score
def evaluate(reg):
    y_pred = reg.predict(x_test)    
    print('score of {} are:  {}'.format(type(reg).__name__, r2_score(y_true, y_pred)))

# training part
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, x_train)
evaluate(lr) # 0.01532419239396543

from sklearn.svm import SVR
svm_svr = SVR()
svm_svr.fit(x_train, y_train)
evaluate(svm_svr) # 0.25819736230712187

from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(x_train, y_train)
evaluate(dt_reg) # -0.5215813506079361

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knn.fit(x_train, y_train)
evaluate(knn) # 0.0989912902325123