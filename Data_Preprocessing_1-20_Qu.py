#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# read data
train_file = '/Users/quzhou/Documents/Kaggle/house-prices-advanced-regression-techniques/train.csv'
df = pd.read_csv(train_file)

# transfer nan into -1
for i in range(1,21):
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
for i in range(1,21):
    temp = df.iloc[:,i]
    names = []
    for j in range(len(temp)):
        if type(temp[j]) == str and temp[j] not in names:
            names.append(temp[j])
    
    for j in range(len(temp)):
        for k in range(len(names)):
            if temp[j] == names[k]:
               temp[j] = k + 1
               break
    df.iloc[:,i] = temp
    
# deal with missing values
# drop colnums with mising values > 50%
for i in range(1,21):
    temp = np.array(df.iloc[:,i])
    index = np.where(temp<0)[0]
    if len(index)/len(temp)>0.5:
        temp[:] = 0
    df.iloc[:,i] = temp

# filling missing values
data0 = df.iloc[:,0:21]

for i in range(1,21):
    
    temp = np.array(df.iloc[:,i])
    index_invalid = np.where(temp<0)[0]
    index_valid = np.where(temp>=0)[0]
    
    if len(index_invalid)==0:
        continue
    x_train = data0.iloc[index_valid,:]
    y_train = x_train.iloc[:,i]
    index = []
    for j in range(1,21):
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
    

data0.to_csv('./preprocessed.csv', index=False, header=True)
     
