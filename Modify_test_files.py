# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:46:14 2022

@author: Ziyi, Qu, and Qihong
"""

import pandas as pd


# read data on win

test_file = '/Users/quzhou/Documents/Kaggle/Kaggle_House_Prices_Advanced_Regression_Techniques-main/test_data_all.csv'
train_file = '/Users/quzhou/Documents/Kaggle/Kaggle_House_Prices_Advanced_Regression_Techniques-main/Processed_data.csv'


df_test = pd.read_csv(test_file)
df_train = pd.read_csv(train_file)

cn_test = df_test.columns
cn_train = df_train.columns

for name_test in cn_test:
    if name_test in cn_train:
        continue
    else:
        df_test = df_test.drop(name_test, axis=1)

for name_train in cn_train:
    if name_train in cn_test:
        continue
    else:
        df_test[name_train] = 0
   

df_all = pd.DataFrame()
for name_train in cn_train:
    df_all[name_train] = df_test[name_train]

##csv
df_all.to_csv('/Users/quzhou/Documents/Kaggle/Kaggle_House_Prices_Advanced_Regression_Techniques-main/test_data_all_final.csv', index=False, header=True)
