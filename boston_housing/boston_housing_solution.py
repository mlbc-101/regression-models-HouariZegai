""" Please inlude the loading, preproessing and the the regression model. """

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
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11) 

# train task
