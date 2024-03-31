import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv(
    r'train.csv')
columns_of_interest = ['LotArea', 'OverallQual', 'YearBuilt','GrLivArea', 'BedroomAbvGr', 'SalePrice']
data = df[columns_of_interest]

data = data.dropna()


X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

linear_regression_model = LinearRegression()
linear_regression_model.fit(X, y)


pickle.dump(linear_regression_model, open('model.pkl', 'wb'))

