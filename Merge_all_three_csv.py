# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:46:14 2022

@author: Ziyi, Qu, and Qihong
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


# read data on win
Qu = 'C:/Users/Qihong/Box/1_Kaggle/Housing_Price_Prediction/preprocessed_Qu(1_20).csv'
Ziyi = 'C:/Users/Qihong/Box/1_Kaggle/Housing_Price_Prediction/preprocessed_Ziyi(21_40).csv'
Qihong = 'C:/Users/Qihong/Box/1_Kaggle/Housing_Price_Prediction/preprocessed_Qihong(41_78).csv'

# Merge 
df_all = pd.concat(map(pd.read_csv, [Qu,Ziyi,Qihong]), axis=1)

##get x and y variables
pr = df_all[['SalePrice']]
df = df_all.drop('SalePrice', axis=1)

##standardize no_dummies columns
no_dummies = [col for col in df.columns if '_' not in col]
df[no_dummies] = StandardScaler().fit_transform(df[no_dummies])

##add y variable back
data0 = pd.concat([df, pr], axis=1)

##csv
data0.to_csv('C:/Users/Qihong/Box/1_Kaggle/Housing_Price_Prediction/Processed_data_all.csv', index=False, header=True)
