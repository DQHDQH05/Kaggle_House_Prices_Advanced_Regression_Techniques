# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 22:16:40 2022

@author: Qihong
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df_path = 'C:/Users/Qihong/Box/1_Kaggle/Housing_Price_Prediction/Processed_data.csv'
df = pd.read_csv(df_path)

# separate train and test
train, test = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv('C:/Users/Qihong/Box/1_Kaggle/Housing_Price_Prediction/train_Qihong.csv', index=False, header=True)
test.to_csv('C:/Users/Qihong/Box/1_Kaggle/Housing_Price_Prediction/test_Qihong.csv', index=False, header=True)