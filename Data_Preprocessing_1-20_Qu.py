#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# read data
train_file = '/Users/quzhou/Documents/Kaggle/house-prices-advanced-regression-techniques/train.csv'
df = pd.read_csv(train_file)
df = df.iloc[:,1:21]
colnums_name = df.columns
# transfer nan into -1
for i in range(len(colnums_name)):
    temp = df.iloc[:,i]
    for j in range(len(temp)):
        if type(temp[j]) == str:
            if temp[j] == 'nan':
                temp[j] = -1
        else:
            if np.isnan(temp[j]):
                temp[j] = -1
    df.iloc[:,i] = temp


# transfer string into numbers
for i in range(len(colnums_name)):
    temp = df.loc[:,colnums_name[i]]
    names = []
    for j in range(len(temp)):
        if type(temp[j]) == str and temp[j] not in names:
            names.append(temp[j])
    if len(names)>0:
        for k in range(len(names)):
            temp_new = temp.copy()
            colnum_name = colnums_name[i] + '_' + names[k]
            for j in range(len(temp)):
                if temp[j] == names[k]:
                   temp_new[j] = 1
                else:
                   temp_new[j] = 0
            df[colnum_name] = temp_new

        df.drop(colnums_name[i], axis=1, inplace=True)
    
# deal with missing values
# drop colnums with mising values > 50%
colnums_name = df.columns
for i in range(len(colnums_name)):
    temp = np.array(df.iloc[:,i])
    index = np.where(temp<0)[0]
    if len(index)/len(temp)>0.5:
        df.drop(colnums_name[i], axis=1, inplace=True)
       

# filling missing values
data0 = df.copy()
colnums_name = df.columns
for i in range(len(colnums_name)):
    
    temp = np.array(df.iloc[:,i])
    index_invalid = np.where(temp<0)[0]
    index_valid = np.where(temp>=0)[0]
    
    if len(index_invalid)==0:
        continue
    x_train = data0.iloc[index_valid,:]
    y_train = x_train.iloc[:,i]
    index = []
    for j in range(len(colnums_name)):
        if i==j:
            continue
        else:
            index.append(j)
    x_train = x_train.iloc[:,index]
    
    x_test = data0.iloc[index_invalid,:]
    x_test = x_test.iloc[:,index]


    knn = KNeighborsRegressor()
    knn.fit(x_train, y_train)
    
    y_pred = knn.predict(x_test)
    data0.iloc[index_invalid,i] = y_pred
    

data0.to_csv('./preprocessed_Qu.csv', index=False, header=True)
     
     
