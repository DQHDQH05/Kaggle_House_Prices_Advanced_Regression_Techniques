# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:02:38 2022

@author: Qihong
"""
conda install -c conda-forge xgboost
pip install xgboost

import pandas as pd # load and manipulate data and for One-Hot Encoding
import numpy as np # calculate the mean and standard deviation
import xgboost as xgb # XGBoost stuff
from sklearn.model_selection import train_test_split # split  data into training and testing sets
from sklearn.model_selection import GridSearchCV # cross validation
from sklearn.metrics import confusion_matrix # creates a confusion matrix
from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix
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

## Check results without tuning
clf_xgb = xgb.XGBRegressor(objective='reg:linear',
                            eval_metric="rmse", ## this avoids a warning...
                            seed=42,
                            use_label_encoder=False)

clf_xgb.fit(X_train, 
             y_train, 
             verbose=True, 
             early_stopping_rounds=10,
             eval_set=[(X_test, y_test)])

#### Hyperparameter tunning
## ROUND 1
# param_grid = {
#     'max_depth': [3, 4, 5],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'gamma': [0, 0.25, 1.0],
#     'reg_lambda': [0, 1.0, 10.0],
#     'reg_alpha': [0, 0.5, 1]
# }
 # ## Output: max_depth: 5, learning_rate: 0.05, gamma: 0, reg_lambda: 0, reg_alpha: 0.5
 # ## Because max_depth, learning_rate and reg_lambda were at the ends of their range, we will continue to explore those...

## ROUND 2
#param_grid = {
#     'max_depth': [5, 6, 7],
#     'learning_rate': [0.03,0.05,0.07],
#     'gamma': [0, 0.05, 0.1],
#     'reg_lambda': [0, 0.05, 0.1],
#     'reg_alpha': [0.3, 0.5, 0.7]
# }
 # ## Output: max_depth: 5, learning_rate: 0.07, gamma: 0, reg_lambda: 0.05, reg_alpha: 0.3

## Round 3
#param_grid = {
#     'max_depth': [5],
#     'learning_rate': [0.07],
#     'gamma': [0],
#     'reg_lambda': [0.03, 0.05, 0.07],
#     'reg_alpha': [0.2, 0.3, 0.4]
# }
# # ## Output: max_depth: 5, learning_rate: 0.07, gamma: 0, reg_lambda: 0.03, reg_alpha: 0.2


 # ## NOTE: To speed up cross validiation, and to further prevent overfitting.
 # ## We are only using a random subset of the data (90%) and are only
 # ## using a random subset of the features (columns) (50%) per tree.
# optimal_params = GridSearchCV(
#      estimator=xgb.XGBRegressor(objective='reg:linear', 
#                                  seed=42,
#                                  subsample=0.9,
#                                  colsample_bytree=0.5,
#                                  early_stopping_rounds=10,
#                                  use_label_encoder=False),
#      param_grid=param_grid,
#      scoring='neg_root_mean_squared_error', ## see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#      verbose=0, # NOTE: If you want to see what Grid Search is doing, set verbose=2
#      n_jobs = 10,
#      cv = 10
# )

# optimal_params.fit(X_train, 
#                    y_train,               
#                    eval_set=[(X_test, y_test)],
#                    verbose=False)
# print(optimal_params.best_params_)


## Final model after tuning
clf_xgb = xgb.XGBRegressor(seed=42,
                        objective='reg:linear',
                        gamma=0,
                        learning_rate=0.07,
                        max_depth=6,
                        reg_lambda=0.03,
                        reg_alpha=0.2,
                        subsample=0.9,
                        colsample_bytree=0.5,
                        use_label_encoder=False)
clf_xgb.fit(X_train, 
            y_train, 
            verbose=True, 
            early_stopping_rounds=10,
            eval_metric="rmse",
            eval_set=[(X_test, y_test)])

# log RMSE 
yhat = clf_xgb.predict(X_test)
rmse = np.sqrt(metrics.mean_squared_error(np.log(y_test), np.log(yhat)))
print("RMSE: %f" % (rmse))

clf_xgb.score(X_test, y_test)

# use Qu Zhou data for final output
df_qu_path = 'C:/Users/Qihong/Box/1_Kaggle/Housing_Price_Prediction/test_data_all_final_new.csv'
df_test_path = 'C:/Users/Qihong/Box/1_Kaggle/Housing_Price_Prediction/test.csv'
df_qu = pd.read_csv(df_qu_path)
df_test = pd.read_csv(df_test_path)
X_qu = df_qu.iloc[:, :-1]
y_qu = df_qu.iloc[:, -1]

# Predicted price
SalePrice = clf_xgb.predict(X_qu)
SalePrice = pd.DataFrame(SalePrice)
SalePrice.columns = ['SalePrice']
Id = df_test['Id']

# prepare dataframe
data0 = pd.concat([Id, SalePrice], axis=1)

##csv
data0.to_csv('C:/Users/Qihong/Box/1_Kaggle/Housing_Price_Prediction/XGBoost_sum.csv', index=False, header=True)
