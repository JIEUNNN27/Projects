#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import nibabel as nib


# In[2]:


#%% dataset
datapath = './' # specify the path where the 'data_regression' directory exists
dataset = ['train', 'test']
datasetxlsfile = os.path.join(datapath, 'Dataset.xlsx')


# In[3]:


#%% variable
variable = {}
variable['response'] = 'Age'
variable['confounding'] = ['TIV', 'Sex']
variable['predictor'] = 'GM'


# In[4]:


#%% atlas
atlas = {}
atlas['imgfile'] = os.path.join(datapath, 'atlas', 'Hammers_60.nii')
v_atlas = nib.load(atlas['imgfile'])
y_atlas = v_atlas.get_fdata()
atlas['labelxlsfile'] = os.path.join(datapath, 'atlas', 'Hammers_60.xls')
atlasT = pd.read_excel(atlas['labelxlsfile'])
atlas['labelidx'] = atlasT['Label'].values


# In[5]:


#%% machinelearningpath
machinelearningpath = os.path.join(datapath, 'machine_learning2')
if not os.path.isdir(machinelearningpath):
    os.makedirs(machinelearningpath)


# In[6]:


#%% xtxtfile
for i_dataset in range(len(dataset)):
    xtxtfile = os.path.join(machinelearningpath, f'X_{dataset[i_dataset]}.txt')
    X = {}
    T = pd.read_excel(datasetxlsfile, sheet_name=dataset[i_dataset])

    # response
    X['response'] = pd.DataFrame()
    if dataset[i_dataset] == 'train':
        if variable['response'] == 'Group':
            X['response'][variable['response']] = T[variable['response']].str.replace('Group', '').astype(int)
        else:
            X['response'][variable['response']] = T[variable['response']]

    # confounding
    X['confounding'] = T[variable['confounding']]

    # predictor
    X['predictor'] = pd.DataFrame()
    for i_subj in range(len(T['Subject'])):
        if variable['response'] == 'Group' and dataset[i_dataset] == 'train':
            subjimgfile = os.path.join(datapath, dataset[i_dataset], f"{T['Group'][i_subj]}_{T['Subject'][i_subj]}_{variable['predictor']}.nii")
        else:
            subjimgfile = os.path.join(datapath, dataset[i_dataset], f"{T['Subject'][i_subj]}_{variable['predictor']}.nii")
        v = nib.load(subjimgfile)
        y = v.get_fdata()
        for i_label in range(len(atlas['labelidx'])):
            X['predictor'].at[i_subj,atlasT['Name'][i_label]] = np.mean(y[y_atlas == atlas['labelidx'][i_label]])
        del subjimgfile, v, y

    # all
    X['all'] = pd.concat([X['response'], X['confounding'], X['predictor']], axis=1)
    X['all'].to_csv(xtxtfile, sep=',', index=False)
    if dataset[i_dataset] == 'train':
        print(f"{dataset[i_dataset]}: {X['all'].shape[0]} samples \u00D7 {X['all'].shape[1]-1} features written")
    elif dataset[i_dataset] == 'test':
        print(f"{dataset[i_dataset]}: {X['all'].shape[0]} samples \u00D7 {X['all'].shape[1]} features written")
    if np.any(np.isnan(X['all'].values)):
        print(f"\t{np.count_nonzero(np.isnan(X['all'].values), axis=1)} samples containing NaNs")
    else:
        print('\tNo samples containing NaNs')
    del xtxtfile, X, T


# In[ ]:




