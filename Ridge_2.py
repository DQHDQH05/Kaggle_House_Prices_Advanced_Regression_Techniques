# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:17:18 2022

@author: Qihong
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics

# read data
df_path = 'C:/Users/Qihong/Box/1_Kaggle/Housing_Price_Prediction/Processed_data.csv'
df = pd.read_csv(df_path)
df.dtypes.unique()

# separate x and y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# separate train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define model evaluation method cross-validation
cv = KFold(n_splits=10)

######################################## ridge regression
# define model
model = Ridge()

# define grid
grid = dict()
grid['alpha'] = np.arange(0, 1, 0.01)

# define search
search = GridSearchCV(model, grid, scoring='r2', cv=cv, n_jobs=-1)

# perform the search
results = search.fit(X,y)

# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

# get accuracy
rid = Ridge(alpha=0.99).fit(X_train, y_train)
yhat = rid.predict(X_test)
print('R2-score: %.3f' % r2_score(np.log(yhat), np.log(y_test)))
print('RMSE: %.3f' % np.sqrt(metrics.mean_squared_error(np.log(yhat), np.log(y_test))))























########################################## ElasticNet
# define model
elastic= ElasticNetCV(l1_ratio = np.arange(0, 1, 0.01), 
                      alphas = [1e-2, 1e-1, 0.0, 1.0, 10.0], 
                      max_iter = 50000, 
                      cv = 10, 
                      tol = 1e-3)
elastic.fit(X,y)
alpha=elastic.alpha_
ratio=elastic.alpha_
print('best alpha:',alpha)
print('best l1 ratio:',ratio)

model = ElasticNet()

# define grid
grid = dict()
grid['alpha'] = [1e-2, 1e-1, 0.0, 1.0, 10.0]
grid['l1_ratio'] = np.arange(0.5, 1, 0.01)

# define search
search = GridSearchCV(model, grid, scoring='r2', cv=cv, n_jobs=-1)

# perform the search
results = search.fit(X, y)

# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

# get accuracy
ela = ElasticNet().fit(X_train, y_train)
yhat = ela.predict(X_test)
print('R2-score: %.2f' % r2_score(yhat, y_test))
