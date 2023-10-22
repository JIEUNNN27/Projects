#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
# from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import GenericUnivariateSelect, f_regression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
from tqdm import tqdm
import lightgbm as lgb


# In[2]:


#%% option
option = {}
option['seed'] = 42
option['applyfeatureselection'] = True
option['thresholdpvalue'] = 0.05
option['cvfoldnum'] = 5
option['cvrepeatnum'] = 1
#option['crange'] = np.logspace(-4,6,num=11)
option['crange'] = np.logspace(-3,3,num=7)


# In[3]:


# 파라미터
param_grid = {
    'n_estimators': [10, 50, 100, 150, 200],  # 트리 개수
    'max_depth': [3, 6, 9, None],  # 트리의 최대 깊이
    'learning_rate': [0.01, 0.1, 0.5],  # 학습률
    'subsample': [0.6, 0.8, 1.0],  # 샘플링 비율
    'colsample_bytree': [0.6, 0.8, 1.0]  # 트리 구성에 사용될 피처 비율
}


# In[4]:


#%% xtxtfile
datapath = './' # specify the path where the 'data_regression' directory exists
machinelearningpath = os.path.join(datapath, 'machine_learning3')
xtxtfile = {}
xtxtfile['train'] = os.path.join(machinelearningpath, 'X_train.txt')
xtxtfile['test'] = os.path.join(machinelearningpath, 'X_test.txt')


# In[5]:


#%% variable
variable = {}
variable['response'] = 'Age'
variable['confounding'] = ['TIV', 'Sex']
variable['predictor'] = 'WM'
variable['categorical'] = ['Sex']


# In[6]:


#%% modelpath
if option['applyfeatureselection']:
    modelpath = os.path.join(machinelearningpath, f"{variable['predictor']}_WithFS_{option['cvrepeatnum']}\u00D7{option['cvfoldnum']}CV")
else:
    modelpath = os.path.join(machinelearningpath, f"{variable['predictor']}_WithoutFS_{option['cvrepeatnum']}\u00D7{option['cvfoldnum']}CV")
if not os.path.exists(modelpath):
    os.makedirs(modelpath)


# In[7]:


#%% X, y
T = pd.read_csv(xtxtfile['train'], delimiter=',')
y_train = T[variable['response']]
X_train = T.drop(columns=[variable['response']])
X_test = pd.read_csv(xtxtfile['test'], delimiter=',')
feature = list(X_test.columns)


# In[8]:


# 진행 상황 표시를 위한 tqdm 객체 생성
progress_bar = tqdm(total=option['cvrepeatnum'] * option['cvfoldnum'])


# In[9]:


#%% summarytxtfile
summarytxtfile = os.path.join(modelpath, 'Summary.txt')
fid = open(summarytxtfile, 'wt')


# In[10]:


#%% cv
cvnum = option['cvfoldnum'] * option['cvrepeatnum']
featureweight = np.full([cvnum, len(feature)],np.nan)
T_featureweight = pd.DataFrame(index=feature)
performance = {"mae": np.empty(cvnum), "rmse": np.empty(cvnum)}
y_predict = np.empty([cvnum, X_test.shape[0]])
random.seed(option['seed'])
for i_repeat in range(option['cvrepeatnum']):
    cv = KFold(n_splits=option['cvfoldnum'], shuffle=True)
    sampleidx = {}
    for i_fold, (sampleidx['train'], sampleidx['test']) in enumerate(cv.split(X_train, y_train)):
        i_cv = (i_repeat * option['cvfoldnum']) + i_fold + 1
        print(f'-------------------- CV {i_cv} --------------------',file=fid)
        progress_bar.update(1)
        
        # X_cv, y_cv
        X_cvtrain, y_cvtrain = X_train.iloc[sampleidx['train']], y_train.iloc[sampleidx['train']]
        X_cvtest, y_cvtest = X_train.iloc[sampleidx['test']], y_train.iloc[sampleidx['test']]
        X_test2 = X_test.copy()

        # sacler
        scaler = StandardScaler()
        X_cvtrain_array = np.empty(X_cvtrain.shape)
        X_cvtest_array = np.empty(X_cvtest.shape)
        X_test2_array = np.empty(X_test2.shape)
        for i_feature in range(len(feature)):
            if feature[i_feature] not in variable['categorical']:
                X_cvtrain_array[:, i_feature] = scaler.fit_transform(X_cvtrain.iloc[:, i_feature].values.reshape(-1, 1)).ravel()
                X_cvtest_array[:, i_feature] = scaler.transform(X_cvtest.iloc[:, i_feature].values.reshape(-1, 1)).ravel()
                X_test2_array[:, i_feature] = scaler.transform(X_test2.iloc[:, i_feature].values.reshape(-1, 1)).ravel()
            else:
                X_cvtrain_array[:, i_feature] = X_cvtrain.iloc[:, i_feature]
                X_cvtest_array[:, i_feature] = X_cvtest.iloc[:, i_feature]
                X_test2_array[:, i_feature] = X_test2.iloc[:, i_feature]
        X_cvtrain = pd.DataFrame(X_cvtrain_array,columns=X_cvtrain.columns,index=X_cvtrain.index)
        X_cvtest = pd.DataFrame(X_cvtest_array,columns=X_cvtest.columns,index=X_cvtest.index)
        X_test2 = pd.DataFrame(X_test2_array,columns=X_test2.columns,index=X_test2.index)
        del scaler, X_cvtrain_array, X_cvtest_array, X_test2_array

        # selector
        if option['applyfeatureselection']:
            selector = GenericUnivariateSelect(f_regression)
            selector.fit(X_cvtrain,y_cvtrain)
            featureidx_selected = selector.pvalues_ <= option['thresholdpvalue']
            X_cvtrain = X_cvtrain.iloc[:,featureidx_selected]
            X_cvtest = X_cvtest.iloc[:,featureidx_selected]
            X_test2 = X_test2.iloc[:,featureidx_selected]
            del selector
        else:
            featureidx_selected = range(len(feature))
        print(f'Training for {X_cvtrain.shape[0]} samples \u00D7 {X_cvtrain.shape[1]} features',file=fid)
        print(f'Test for {X_cvtest.shape[0]} samples \u00D7 {X_cvtest.shape[1]} features',file=fid)

        # xgboost
        mdl = lgb.LGBMRegressor()
        searcher = GridSearchCV(mdl, param_grid=param_grid, scoring='neg_mean_absolute_error')
        searcher.fit(X_cvtrain, y_cvtrain)
        mdl = searcher.best_estimator_
        #featureweight[i_cv-1, featureidx_selected] = mdl.coef_
        T_featureweight[f"CV{i_cv}"] = featureweight[i_cv-1, :]
        performance["mae"][i_cv-1] = mean_absolute_error(y_cvtest, mdl.predict(X_cvtest))
        performance["rmse"][i_cv-1] = mean_squared_error(y_cvtest, mdl.predict(X_cvtest))**0.5
        y_predict[i_cv-1, :] = mdl.predict(X_test2)
        print(f"MAE = {performance['mae'][i_cv-1]:.3f}",file=fid)
        print(f"RMSE = {performance['rmse'][i_cv-1]:.3f}",file=fid)
        del i_cv, X_cvtrain, y_cvtrain, X_cvtest, y_cvtest, X_test2, featureidx_selected, searcher, mdl
    del cv, sampleidx

#%% avearage
print('-------------------- Average --------------------',file=fid)
featureweight_average = np.mean(featureweight, axis=0)
T_featureweight['Average'] = featureweight_average
T_featureweight.to_excel(os.path.join(modelpath, 'FeatureWeight.xlsx'), index=True)
featureidx_common = ~np.isnan(featureweight_average)
print(f'{np.count_nonzero(featureidx_common)}/{len(feature)} features commonly used',file=fid)
print(f"MAE = {np.mean(performance['mae']):.3f}\u00B1{np.std(performance['mae']):.3f}",file=fid)
print(f"RMSE = {np.mean(performance['rmse']):.3f}\u00B1{np.std(performance['rmse']):.3f}",file=fid)
np.savetxt(os.path.join(modelpath, 'y_predict.txt'), y_predict)
fid.close()
progress_bar.close()


# In[11]:


y_predict_df = pd.DataFrame(y_predict)


# In[12]:


y_predict_df = pd.DataFrame(y_predict)


# In[13]:


y_predict_df.to_csv('./y_predict_WM_LightGBM.csv')


# In[ ]:




