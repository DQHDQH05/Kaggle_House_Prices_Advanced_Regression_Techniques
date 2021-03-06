import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# read data
train_file = 'C:/Users/Qihong/Box/1_Kaggle/Housing_Price_Prediction/train.csv'
df_o = pd.read_csv(train_file)


pr=df_o.iloc[:, 80]#price
df=df_o.iloc[:, 21:41]#characteristics


#Check if some nan are meaningful
df.isnull().sum()
##Seems NA represent No_Bsmt
Bsmt = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']
df[Bsmt] = df[Bsmt].replace(pd.NA, 'No_Bsmt')

#1 Delete columns if >50% NaN 
df = df.loc[:, (df.isnull().sum(axis=0) <= len(pr)/2)]  #Delete features if >50% NaN 

#2 Convert string to number (dummies)
#df[df.columns[5]].dtype type
for i in range(0,20):
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



data0.to_csv('C:/Users/Qihong/Box/1_Kaggle/Housing_Price_Prediction/preprocessed_Ziyi(21_40).csv', index=False, header=True)
     
