# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:41:21 2022

@author: Qihong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"C:\Users\Qihong\Box\1_Kaggle\Housing_Price_Prediction\train.csv")
df.duplicated().sum()

##Some dataframe regular checks
df4 = df.iloc[:,58:]
df4.shape
df4.info
df4.dtypes
df4.isnull().sum()

##delte Column: PoolQC, Fence, MiscFeature
df4 = df4.drop(['PoolQC','Fence','MiscFeature'],1)

##Check numerical feature relationships
sns.pairplot(df4)
plt.show()
plt.savefig('60-80variables.png')

##Handle missing value in GarageYrBlt
median_GarageYrBlt=df4['GarageYrBlt'].median()
df4['GarageYrBlt'].fillna(median_GarageYrBlt,inplace=True)

##Rename all no garage cell
df4 = df4.replace(pd.NA, 'No_g')

##Check relationships between 'object' features and price
sns.catplot(y="SalePrice", x="GarageType", data=df4)
plt.show()
sns.catplot(y="SalePrice", x="GarageFinish", data=df4)
plt.show()
sns.catplot(y="SalePrice", x="GarageQual", data=df4)
plt.show()
sns.catplot(y="SalePrice", x="GarageCond", data=df4)
plt.show()
sns.catplot(y="SalePrice", x="PavedDrive", data=df4)
plt.show()
