#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:35:29 2023

@author: WBR
"""

#%%

# M4 - Add predictor (fb_time) for the amount of time spent post-response on the same problem
# (e.g., processing correct/incorrect feedback)
# This predictor necessarily is confounded by item difficulty, therefore not totally surprising that model did not improve
    # Q: What is the relationship between this measure and gaming-the-system?

#%%
import pandas as pd
import numpy as np
from numpy import nanmean
from helplearn.dataframe import peek_df
from helplearn.dataframe import proportion_nans
from sklearn.linear_model import LogisticRegression
from datetime import datetime
#%%  import data

ddir = '/Users/WBR/walter/local_professional/EDM_Cup/edm-cup-2023/'

training_unit_test_scores = pd.read_csv(ddir + 'training_unit_test_scores.csv')
evaluation_unit_test_scores = pd.read_csv(ddir + 'evaluation_unit_test_scores.csv')
assignment_details = pd.read_csv(ddir + 'assignment_details.csv')
action_logs = pd.read_csv(ddir + 'action_logs.csv')
assignment_relationships = pd.read_csv(ddir + 'assignment_relationships.csv')
problem_details = pd.read_csv(ddir + 'problem_details.csv')
euts = pd.read_csv(ddir + 'euts_m2.csv')
tuts = pd.read_csv(ddir + 'tuts_m2.csv')

# prepending "unit_test_" to assignment_log_id for clarity and to dovetail with assignment_relationships.csv naming
training_unit_test_scores.columns = ['unit_test_assignment_log_id', 'problem_id', 'score']
evaluation_unit_test_scores.columns = ['id', 'unit_test_assignment_log_id', 'problem_id', 'score']

# convert timestamp
action_logs['timestamp'] = pd.to_datetime(action_logs['timestamp'], unit='s')

# drop assignment level c_prop
# tuts.drop('mean_c_prop',inplace=True,axis=1)
# euts.drop('mean_c_prop',inplace=True,axis=1)

# put all the column names in one place 
column_names = {'training_unit_test_scores':training_unit_test_scores.columns,
           'evaluation_unit_test_scores':evaluation_unit_test_scores.columns,
           'assignment_details':assignment_details.columns,
           'action_logs':action_logs.columns,
           'assignment_relationships':assignment_relationships.columns,
           'problem_details':problem_details.columns,
           'tuts':tuts.columns,
           'euts':euts.columns}
# and make it nice to look at
column_names = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in column_names.items() ]))

#%% Make fb_time

act_time = action_logs[['assignment_log_id','timestamp','problem_id','action']].copy()
act_time = pd.get_dummies(act_time, columns=['action'])
act_time = act_time[['assignment_log_id','timestamp','problem_id','action_correct_response','action_wrong_response','action_problem_finished']]

# 1 for first response on a problem
act_time['p_start'] = act_time['action_correct_response'] + act_time['action_wrong_response']

# keep only needed rows
act_time = act_time[(act_time['p_start'] ==1) | (act_time['action_problem_finished'] == 1)]

# keep only first response action whether correct or not
act_time = act_time.sort_values('timestamp').drop_duplicates(subset=['assignment_log_id','problem_id','p_start'])

# calculate elapsed time
act_time['p_diff'] = act_time.groupby(['assignment_log_id','problem_id'])['timestamp'].transform('diff')
# check time dtype

# drop NATs
act_time.dropna(inplace=True)

# convert from Timedelta dtype to seconds
act_time['p_diff_float'] = act_time['p_diff'].dt.total_seconds()

# log transform
act_time['p_diff_log'] = np.log(act_time['p_diff_float'])

# final col select
act_time = act_time[['assignment_log_id','p_diff_log','problem_id']]

#%% merge 

# add problem_skill_code
pr_d = act_time.merge(problem_details[['problem_id','problem_skill_code']],how='left',on='problem_id')

pr_d['mean_skill_act_time'] = pr_d.groupby(['assignment_log_id','problem_skill_code'])['p_diff_log'].transform('mean')

# drop duplicates
pr_d.drop_duplicates(subset=['assignment_log_id','problem_skill_code'],inplace=True)

# add unit_test_assignment_log_id
ar = assignment_relationships.merge(pr_d,left_on='in_unit_assignment_log_id',right_on='assignment_log_id')

# average by unit_test log and skill code
ar = ar.groupby(['unit_test_assignment_log_id','problem_skill_code'])['mean_skill_act_time'].mean().reset_index()

# add problem_skill_code to test scores
tuts = tuts.merge(problem_details[['problem_id','problem_skill_code']],how='left',on='problem_id')
euts = euts.merge(problem_details[['problem_id','problem_skill_code']],how='left',on='problem_id')

tuts = tuts.merge(ar,how='left',on=['unit_test_assignment_log_id','problem_skill_code'])
euts = euts.merge(ar,how='left',on=['unit_test_assignment_log_id','problem_skill_code'])


#%%

# fill NA, would be better to fill with subject mean
# tuts['mean_skill_act_time'].fillna(tuts['mean_skill_act_time'].mean(), inplace=True)
# euts['mean_skill_act_time'].fillna(euts['mean_skill_act_time'].mean(), inplace=True)
# m5.2, fillna with 0 instead of mean
tuts['mean_skill_act_time'].fillna(0, inplace=True)
euts['mean_skill_act_time'].fillna(0, inplace=True)
euts['target_encoded_skill'].fillna(euts['target_encoded_skill'].mean(), inplace=True)
# # for m4 we added back overall sub skill
# tuts['mean_c_prop'].fillna(tuts['mean_c_prop'].mean(), inplace=True)
# euts['mean_c_prop'].fillna(euts['mean_c_prop'].mean(), inplace=True)

proportion_nans(tuts)

#%% M5

input_cols = ['mean_c_prop','target_encoded_skill','mean_skill_act_time']
target_col = 'score'

# Initialize a logistic regression
lr = LogisticRegression(max_iter=1000)
# Fit the regression on all the training data
lr = lr.fit(tuts[input_cols], tuts[target_col])

# Predict the score for each evaluation problem
euts[target_col] = lr.predict(euts[input_cols])

euts[['id', 'score']].to_csv(ddir + 'm5.2.csv', index=False)

#%%

















#%%
#%%
#%%


















#%%
#%%
#%%




















