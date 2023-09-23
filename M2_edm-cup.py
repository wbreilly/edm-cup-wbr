#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:44:06 2023

@author: WBR
"""

#%%
import pandas as pd
import numpy as np
from numpy import nanmean
from helplearn.dataframe import peek_df
from helplearn.dataframe import proportion_nans
from sklearn.linear_model import LogisticRegression
#%% Notes on data sources

# you are tasked with predicting students’ scores on end-of-unit assignment problems,
# given their click-stream data across all the in-unit assignments they completed previously.

# *_unit_test_scores are the outcome to train on and predict
# assignment log ID corresponds to a particular instance of a student starting a problem set assigned to them
# action_logs.csv does not contain subject info. Does contain clickstream.
# assignment and sequence relationships - inform which information from action_logs.csv corresponds to each unit test score that must be predicted
#%%  import data

ddir = '/Users/WBR/walter/local_professional/EDM_Cup/edm-cup-2023/'

training_unit_test_scores = pd.read_csv(ddir + 'training_unit_test_scores.csv')
evaluation_unit_test_scores = pd.read_csv(ddir + 'evaluation_unit_test_scores.csv')
assignment_details = pd.read_csv(ddir + 'assignment_details.csv')
action_logs = pd.read_csv(ddir + 'action_logs.csv')
assignment_relationships = pd.read_csv(ddir + 'assignment_relationships.csv')
problem_details = pd.read_csv(ddir + 'problem_details.csv')

# prepending "unit_test_" to assignment_log_id for clarity and to dovetail with assignment_relationships.csv naming
training_unit_test_scores.columns = ['unit_test_assignment_log_id', 'problem_id', 'score']
evaluation_unit_test_scores.columns = ['id', 'unit_test_assignment_log_id', 'problem_id', 'score']

# put all the column names in one place 
column_names = {'training_unit_test_scores':training_unit_test_scores.columns,
           'evaluation_unit_test_scores':evaluation_unit_test_scores.columns,
           'assignment_details':assignment_details.columns,
           'action_logs':action_logs.columns,
           'assignment_relationships':assignment_relationships.columns,
           'problem_details':problem_details.columns}
# and make it nice to look at
column_names = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in column_names.items() ]))

#%% target encode problem_skill_codes

pr_d = training_unit_test_scores.merge(problem_details[['problem_id','problem_skill_code']],how='left',on='problem_id')

pr_d['target_encoded_skill'] = pr_d.groupby('problem_skill_code')['score'].transform('mean')
pd.isna(pr_d['problem_skill_code']).sum() # about 9% has no skill code 

# is it better to drop rows or fill na with mean? try both
# pr_d.dropna(inplace=True) # doesn't work because rows in evaluation set need value
# prd_d_fill = pr_d['targe_encoded_skill'].fillna()
pr_d['target_encoded_skill'].fillna(pr_d['target_encoded_skill'].mean(), inplace=True)

# drop unnecessary columns and rows
pr_d = pr_d[['problem_id','target_encoded_skill']].drop_duplicates().copy()
#%% read tuts and euts 

tuts = pd.read_csv(ddir + 'tuts.csv')
euts = pd.read_csv(ddir + 'euts.csv')

# tuts.to_csv(ddir + 'tuts_m2.csv',index=False)
# euts.to_csv(ddir + 'euts_m2.csv',index=False)

#%% merge scores with problem details

tuts = tuts.merge(pr_d[['problem_id','target_encoded_skill']],how='left',on='problem_id')
euts = euts.merge(pr_d[['problem_id','target_encoded_skill']],how='left',on='problem_id')

# fill NA
tuts['mean_c_prop'].fillna(tuts['mean_c_prop'].mean(), inplace=True)
euts['mean_c_prop'].fillna(euts['mean_c_prop'].mean(), inplace=True)
euts['target_encoded_skill'].fillna(euts['target_encoded_skill'].mean(), inplace=True)

proportion_nans(tuts)

#%% M2

input_cols = ['mean_c_prop','target_encoded_skill']
target_col = 'score'

# Initialize a logistic regression
lr = LogisticRegression(max_iter=1000)
# Fit the regression on all the training data
lr = lr.fit(tuts[input_cols], tuts[target_col])

# Predict the score for each evaluation problem
euts[target_col] = lr.predict(euts[input_cols])

euts[['id', 'score']].to_csv(ddir + 'm2.csv', index=False)
#%%