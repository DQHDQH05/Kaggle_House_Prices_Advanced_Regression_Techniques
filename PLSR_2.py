#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 22:44:42 2022

@author: aaronli
"""
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
import numpy as np # for array operations
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from scipy.stats import pearsonr

from scipy import optimize

def f_1(x, A, B):
    return A*x + B

def density_scatter( x, y, tle, ax = None, sort = True, bins = 20, **kwargs)   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    ax.set_xlabel( 'Log (Actual values ($))', fontsize=12)
    ax.set_ylabel('Log (Predicted values ($))', fontsize=12)
    x1 = np.arange(0,14, 0.01)
    #y1 = A1*x1 + B1
    A1, B1 = optimize.curve_fit(f_1, x, y)[0]
    y1 = A1*x1 + B1
    
    r1,p1= pearsonr(x,y)
    r1=round(r1, 1) 

    rms = sqrt(mean_squared_error(x,y))
    
    plt.plot(x1, x1, "black",ls='--',label='1:1')  
    
    if B1>0:
        ax.plot(x1, y1, "purple",ls='--',label='Y='+str(round(A1,1))+'X+'+str(round(B1,1)))
    else:
        ax.plot(x1, y1, "purple",ls='--',label='Y='+str(round(A1,1))+'X'+str(round(B1,1)))

    ax.text(12.5, 10, '$\mathregular{R^{2}}$='+str(round(r2_score(x,y),2))+'\nRMSE='+str(round(np.sqrt(metrics.mean_squared_error(x,y)),2)), fontsize=10)
    ax.set_title(tle,fontsize=17) 
    ax.legend(fontsize=10,frameon=False,loc='upper left')
    ax.set_xlim([9.5, 14])
    ax.set_ylim([9.5, 14])

    return ax




# load data
df_path = '/Users/aaronli/Documents/Course/kaggle/house_price/Model/Processed_data.csv'
df = pd.read_csv(df_path)
x=df.iloc[:,0:len(df.columns)-1].values
x1=df.iloc[:,0:len(df.columns)-1].values

y=df.iloc[:, -1]#price

# separate train and test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#define cross-validation
cv=KFold(n_splits=10)

#define model
model = PLSRegression()

#define grid
grid=dict()
grid['n_components']=np.arange(1,50,1)

#define search
search=GridSearchCV(model,grid,scoring='r2',cv=cv,n_jobs=-1)

#perform the search
results=search.fit(x_train, y_train)

#summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

#accuracy
PL=PLSRegression(n_components = results.best_params_.get('n_components')).fit(x_train,y_train)
yrf=PL.predict(x_test)

print('R2-score: %.3f' % r2_score(np.log(yrf),np.log(y_test)))
print('MSE: %.3f' % np.sqrt(metrics.mean_squared_error(np.log(yrf),np.log(y_test))))


#calibration
yrft=PL.predict(x_train)
density_scatter(np.log(y_train.values),np.log(yrft.flatten()), tle='PLSR',bins = [30,30])

#validation
density_scatter(np.log(y_test.values),np.log(yrf.flatten()), tle='PLSR',bins = [30,30])

#yrf prediction value

#density_scatter(np.log(yrf),np.log(y_test), tle='PLSR',bins = [30,30])


#print('R2-score: %.3f' % r2_score(yrf,y_test))
#print('MSE: %.3f' % np.sqrt(metrics.mean_squared_error(yrf,y_test)))



#train.to_csv('/Users/aaronli/Documents/Course/kaggle/house_price/Model/train_Qihong.csv', index=False, header=True)
#test.to_csv('/Users/aaronli/Documents/Course/kaggle/house_price/Model/test_Qihong.csv', index=False, header=True)

