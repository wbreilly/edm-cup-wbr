#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:00:32 2023

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
euts = pd.read_csv(ddir + 'euts_m2.csv')
tuts = pd.read_csv(ddir + 'tuts_m2.csv')

# prepending "unit_test_" to assignment_log_id for clarity and to dovetail with assignment_relationships.csv naming
training_unit_test_scores.columns = ['unit_test_assignment_log_id', 'problem_id', 'score']
evaluation_unit_test_scores.columns = ['id', 'unit_test_assignment_log_id', 'problem_id', 'score']

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

#%% M3: proportion correct actions at problem_skill_code level (unlike m1 which was at assignment level)

# start by getting sum of actions taken per problem per assignment assignment
actions = pd.get_dummies(action_logs, columns=['action'])
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
actions_sum.columns

#%%

# add problem details to then compute mean skill code for each participant
pr_d = actions_sum.merge(problem_details[['problem_id','problem_skill_code']],how='right',on='problem_id')
pr_d['mean_c_sub_skill_code'] = pr_d.groupby(['assignment_log_id','problem_skill_code']).transform('mean')
propnan = proportion_nans(pr_d)

# add problem_skill_code to test scores
tuts = tuts.merge(problem_details[['problem_id','problem_skill_code']],how='left',on='problem_id')
euts = euts.merge(problem_details[['problem_id','problem_skill_code']],how='left',on='problem_id')

# add unit_test_assignment_log_id
ar = assignment_relationships.merge(pr_d,left_on='in_unit_assignment_log_id',right_on='assignment_log_id')
ar = ar.drop_duplicates(subset =['assignment_log_id','problem_skill_code','mean_c_sub_skill_code'])
# average by unit_test log and skill code
ar = ar.groupby(['unit_test_assignment_log_id','problem_skill_code'])['mean_c_sub_skill_code'].mean().reset_index()

tuts = tuts.merge(ar,how='left',on=['unit_test_assignment_log_id','problem_skill_code'])
euts = euts.merge(ar,how='left',on=['unit_test_assignment_log_id','problem_skill_code'])

#%%

# fill NA, would be better to fill with subject mean
tuts['mean_c_sub_skill_code'].fillna(tuts['mean_c_sub_skill_code'].mean(), inplace=True)
euts['mean_c_sub_skill_code'].fillna(euts['mean_c_sub_skill_code'].mean(), inplace=True)
euts['target_encoded_skill'].fillna(euts['target_encoded_skill'].mean(), inplace=True)
# for m4 we added back overall sub skill
tuts['mean_c_prop'].fillna(tuts['mean_c_prop'].mean(), inplace=True)
euts['mean_c_prop'].fillna(euts['mean_c_prop'].mean(), inplace=True)

proportion_nans(euts)

#%% M3/M4

input_cols = ['mean_c_sub_skill_code','mean_c_prop','target_encoded_skill']
target_col = 'score'

# Initialize a logistic regression
lr = LogisticRegression(max_iter=1000)
# Fit the regression on all the training data
lr = lr.fit(tuts[input_cols], tuts[target_col])

# Predict the score for each evaluation problem
euts[target_col] = lr.predict(euts[input_cols])

euts[['id', 'score']].to_csv(ddir + 'm4.csv', index=False)
#%%
#%%
#%%
#%%
















