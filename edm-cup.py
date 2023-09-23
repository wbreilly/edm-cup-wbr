#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:59:17 2023

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


#%% Prepare data for M1, predict score based on assignment practice performance

# accuracy will be prop of correct response at subject level
# start by getting sum of actions taken per assignment
actions = pd.get_dummies(action_logs, columns=['action'])
actions_sum = actions.groupby('assignment_log_id').sum()

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

# merge pertinent student info into actions_sum
# select relevant colummns from assignment_details
ad = assignment_details[['assignment_log_id','student_id']].copy()
# merge with actions_sum
ad = ad.merge(actions_sum,on='assignment_log_id')
ad.columns
len(ad) == len(actions_sum) # good merge

# compute mean correct at student level
# ad['mean_prop_correct'] = ad.groupby('student_id')['c_prop'].transform('mean')

# link student assignment score to unit score with assignment_relationships
ar = assignment_relationships.merge(ad,left_on='in_unit_assignment_log_id',right_on='assignment_log_id')

# link to training/test unit score
tuts = training_unit_test_scores.merge(ar, how='inner', on='unit_test_assignment_log_id')

### new version
tuts['mean_c_prop'] = tuts.groupby(['unit_test_assignment_log_id','problem_id'])['c_prop'].transform('mean')
tuts = tuts[['unit_test_assignment_log_id','problem_id','score','mean_c_prop']].drop_duplicates().copy()
###

len(training_unit_test_scores)
len(ar)
len(tuts)
tuts.isna().sum()
peek = peek_df(tuts,1000)
tuts.student_id.nunique() 
tuts.unit_test_assignment_log_id.nunique()
check = tuts.groupby('student_id')['unit_test_assignment_log_id'].nunique() 
check.describe() # mean 1.6 unit_tests per student. Max 15
check = tuts.groupby('student_id')['assignment_log_id'].nunique() 
check.describe() # mean 10.5 practice sets per student. Max 193?! #That person likley logging on and off repeatedly, should remove

# Merge action count features with the evaluation unit test scores
euts = evaluation_unit_test_scores.merge(ar, how='left', on='unit_test_assignment_log_id')
### new version
euts['mean_c_prop'] = euts.groupby('id')['c_prop'].transform(nanmean)
euts = euts[['id','problem_id','mean_c_prop','unit_test_assignment_log_id']].drop_duplicates().copy()
###

# euts.student_id.nunique() 
euts.unit_test_assignment_log_id.nunique()
# check = euts.groupby('student_id')['unit_test_assignment_log_id'].nunique() 
# check.describe() # mean 1.4 unit_tests per student. Max 14

tuts.to_csv(ddir + 'tuts.csv',index=False)
euts.to_csv(ddir + 'euts.csv',index=False)
#%% Model 1

# tuts needs to have predictors of same length as score
# we have a score for each problem on an end-of-unit assignment
# Therefore predictor should be student-level performance on similar practice problems
# But what's given is a mapping between practice sets and end of unit sets
# Therefore simplest model will ignore problem level 

# sklearn needs numeric variables
# not an option to use categorical predictors due to memory demand of one-hot encoding

input_cols = ['mean_c_prop']
target_col = 'score'

tuts.dropna(inplace=True)
euts.fillna(0,inplace=True)

# Initialize a logistic regression
lr = LogisticRegression(max_iter=1000)
# Fit the regression on all the training data
lr = lr.fit(tuts[input_cols], tuts[target_col])

# Predict the score for each evaluation problem
# euts[target_col] = lr.predict_proba(euts[input_cols])[:,1]
euts[target_col] = lr.predict(euts[['mean_c_prop']])

euts[['id', 'score']].to_csv(ddir + 'm1.csv', index=False)

peek = peek_df(euts,1000)
#%%