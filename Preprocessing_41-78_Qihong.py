# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:41:21 2022

@author: Qihong
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

##read file
df=pd.read_csv(r"C:\Users\Qihong\Box\1_Kaggle\Housing_Price_Prediction\train.csv")
df.duplicated().sum()


##Some dataframe regular checks
df = df.iloc[:,42:]
df.shape
df.info
df.dtypes
df.isnull().sum()


#Check if some nan are meaningful
##Rename all nan garage cell
garage = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
df[garage] = df[garage].replace(pd.NA, 'No_garage')
##Rename all nan FireplaceQu
df['FireplaceQu'] = df['FireplaceQu'].replace(pd.NA, 'No_fireplace')


#1 Delete columns if >50% NaN 
##delte Column: PoolQC, Fence, MiscFeature
df = df.loc[:, (df.isnull().sum(axis=0) <= len(df)/2)] #Delete features if >50% NaN 


#2 Convert string to number (dummies)
#df[df.columns[5]].dtype type
for i in range(len(df.columns)):
    temp=df[df.columns[i]]


    if temp.dtype=='O': #string
        A=pd.get_dummies(temp).rename(columns=lambda x:temp.name +'_'+str(x))
    
    else:
        A=temp
        
    if i==0:
        dfn=A
    else:
        dfn=pd.concat([dfn, A], axis=1)


#3 KNN regression to fill resting features with NaNs based on other value

# 3.1 transfer nan into -1
for i in range(0,len(dfn.columns)):
    temp = dfn.iloc[:,i]
    for j in range(len(temp)):
        if type(temp[j]) == str:
            if temp[j] == 'nan':
                temp[j] = -1
        else:
            if np.isnan(temp[j]):
                temp[j] = -1
    dfn.iloc[:,i] = temp
    
    

data0 = dfn



for i in range(0,len(dfn.columns)):
    
    
    temp = np.array(dfn.iloc[:,i])
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



data0.to_csv('C:/Users/Qihong/Box/1_Kaggle/Housing_Price_Prediction/preprocessed_Qihong(61_80).csv', index=False, header=True)