#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:27:37 2023

@author: WBR
"""


#%%

# M6 - make each problem_skill_code performance a distinct predictor
# This will have the same information, but advantage is that different skill codes can have different weights
# 541 unique problem_skill_code - hopefully feasible for laptop to run
# Next model should have unique predictor for each problem – but this would require 132738 new predictors 

#%%
import pandas as pd
import numpy as np
from numpy import nanmean
from helplearn.dataframe import peek_df
from helplearn.dataframe import proportion_nans
from helplearn.data import load_dataset
from helplearn.data import get_column_names
from sklearn.linear_model import LogisticRegression
from datetime import datetime

#%%  import data

ddir = '/Users/WBR/walter/local_professional/EDM_Cup/edm-cup-2023/'

data, column_names = load_dataset(ddir)

# prepending "unit_test_" to assignment_log_id for clarity and to dovetail with assignment_relationships.csv naming
data['training_unit_test_scores'].columns = ['unit_test_assignment_log_id', 'problem_id', 'score']
data['evaluation_unit_test_scores'].columns = ['id', 'unit_test_assignment_log_id', 'problem_id', 'score']

# convert timestamp
data['action_logs']['timestamp'] = pd.to_datetime(data['action_logs']['timestamp'], unit='s')

# update column_names
column_names = get_column_names(data)

#%% from M3: proportion correct actions at problem_skill_code level (unlike m1 which was at assignment level)

# start by getting sum of actions taken per problem per assignment assignment
actions = pd.get_dummies(data['action_logs'], columns=['action'])
actions_sum = actions.groupby(['assignment_log_id','problem_id']).sum()

# compute correct proportion 
actions_sum['c_prop'] = actions_sum['action_correct_response'] / (actions_sum['action_correct_response'] + actions_sum['action_wrong_response'])
# prop step is introducing nans

# keep only necessary columns
actions_sum = actions_sum[['action_correct_response','action_wrong_response','c_prop']].reset_index()
# nans appear to result  from assignment_log_ids that have 0s 
# remove nans (thereby also uninformative assignment_log_ids)
actions_sum = actions_sum[((actions_sum['action_correct_response'] != 0) & 
                           (actions_sum['action_wrong_response'] != 0))]
nancount = pd.isna(actions_sum).sum() # checks out
# drop unneeded columns again
actions_sum = actions_sum.drop(['action_correct_response','action_wrong_response'],axis=1) 

#%%  from M3 as well 

pr_d = actions_sum.merge(data['problem_details'][['problem_id','problem_skill_code']],how='right',on='problem_id')
 
pr_d['mean_c_sub_skill_code'] = pr_d.groupby(['assignment_log_id','problem_skill_code']).transform('mean')
propnan = proportion_nans(pr_d)

# add problem_skill_code to test scores
tuts = data['tuts'].merge(data['problem_details'][['problem_id','problem_skill_code']],how='left',on='problem_id')
euts = data['euts'].merge(data['problem_details'][['problem_id','problem_skill_code']],how='left',on='problem_id')

# add unit_test_assignment_log_id
ar = data['assignment_relationships'].merge(pr_d,left_on='in_unit_assignment_log_id',right_on='assignment_log_id')
ar = ar.drop_duplicates(subset =['assignment_log_id','problem_skill_code','mean_c_sub_skill_code'])
# average by unit_test log and skill code
ar = ar.groupby(['unit_test_assignment_log_id','problem_skill_code'])['mean_c_sub_skill_code'].mean().reset_index()

tuts = tuts.merge(ar,how='left',on=['unit_test_assignment_log_id','problem_skill_code'])
euts = euts.merge(ar,how='left',on=['unit_test_assignment_log_id','problem_skill_code'])

#%% check if same problem_skill_code in tuts and euts

psc_tuts = set(tuts.problem_skill_code.unique())
psc_euts = set(euts.problem_skill_code.unique())

difference = psc_tuts ^ psc_euts 

missing_euts = psc_tuts - psc_euts
missing_tuts = psc_euts - psc_tuts
# 37/335 or 11%

# make provblem_skill_code wide format
tuts_pivot = tuts.pivot(index=['unit_test_assignment_log_id','problem_id'],columns = 'problem_skill_code',values='mean_c_sub_skill_code')
tuts_pivot.fillna(0,inplace=True)
tuts_pivot.reset_index(inplace=True)
# add columns for missing problem_skill_codes
tuts = tuts.merge(tuts_pivot,on=['unit_test_assignment_log_id','problem_id'])

euts_pivot = euts.pivot(index=['unit_test_assignment_log_id','problem_id'],columns = 'problem_skill_code',values='mean_c_sub_skill_code')
euts_pivot.fillna(0,inplace=True)
euts_pivot.reset_index(inplace=True)
# add columns for missing problem_skill_codes
euts = euts.merge(euts_pivot,on=['unit_test_assignment_log_id','problem_id'])

# now make columns identical between tuts and euts
for col_name in missing_euts:
    euts[col_name] = 0
    
for col_name in missing_tuts:
    tuts[col_name] = 0

# check that col_names are identical
euts_columns = set(euts.columns)
tuts_columns = set(tuts.columns)

difference = euts_columns ^ tuts_columns

# add back score
euts['score'] = np.nan

#%% fillna

# target_encoded_skill is missing
eutss = pd.read_csv(ddir + 'euts_m2.csv')
tutss = pd.read_csv(ddir + 'tuts_m2.csv')

euts = euts.merge(eutss[['problem_id','target_encoded_skill']].drop_duplicates(),on='problem_id') 
euts['target_encoded_skill'].fillna(euts['target_encoded_skill'].mean(), inplace=True)

tuts = tuts.merge(eutss[['problem_id','target_encoded_skill']].drop_duplicates(),on='problem_id') 
tuts['target_encoded_skill'].fillna(tuts['target_encoded_skill'].mean(), inplace=True)

tuts['mean_c_prop'].fillna(tuts['mean_c_prop'].mean(), inplace=True)
euts['mean_c_prop'].fillna(euts['mean_c_prop'].mean(), inplace=True)

#%% M6
 

all_cols = tuts.columns
drop_cols = ['unit_test_assignment_log_id','score','problem_skill_code','problem_id','mean_c_sub_skill_code',np.nan]
input_cols = list(set(all_cols) - set(drop_cols))
input_cols = [str(x) for x in input_cols]

target_col = 'score'

# Initialize a logistic regression
lr = LogisticRegression(max_iter=1000)
# Fit the regression on all the training data
lr = lr.fit(tuts[input_cols], tuts[target_col])

# Predict the score for each evaluation problem
euts[target_col] = lr.predict(euts[input_cols])

euts[['id', 'score']].to_csv(ddir + 'm6.csv', index=False)
#%%
 
#%%
 
#%%
 
#%%





















 