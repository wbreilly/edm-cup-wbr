#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:35:47 2023

@author: WBR
"""

# data prep for LKT in R.

# LKT follows DataShop format
# Critical is that actions be in consecutive order grouped by learner id

# will want at least correct and incorrect actions from action_logs
# problem_skill_code from problem details
# student_id from assignment_details
# unit_test_assigment_log_id from assignment_relationships 

# once models are estimated for each KC for each student, should be able to predict unit_test scores yeah? 

# how many problems per problem_skill_code? # 243

check = pr_d[['problem_id','problem_skill_code']].drop_duplicates()
check.groupby('problem_skill_code')['problem_id'].count().mean() # 243

# I guess what's strange is that we know future performance for training set. 
# are we trying to predict performance of brand new students? No
# So yeah this problem is more about discovering new features from clickstream...
# not so much finding right model spec with vanilla predictors
# Nevertheless, I should learn to use LKT

# IMPORTANT, can't estimate KC for each student in python becuase of large N and data size
# I believe that R does better with data size using sparse matrices or something
 

#%%
import pandas as pd
import numpy as np
from numpy import nanmean
from helplearn.dataframe import peek_df
from helplearn.dataframe import proportion_nans
from helplearn.data import load_dataset
from helplearn.data import get_column_names
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

#%% action_logs : whittle down to correct, incorrect actions, timestamp, and id info

actions = data['action_logs'][['assignment_log_id','timestamp','problem_id','action']].copy()

actions = actions[(actions['action'] == 'correct_response') | (actions['action'] == 'wrong_response')]
actions = pd.get_dummies(actions,columns=['action'])
actions = actions.drop(columns=['action_wrong_response'])

# peek = peek_df(act_time,1000)

#%% whittle actions further down to first attempts only

if first_attempts:
    act_time = data['action_logs'][['assignment_log_id','timestamp','problem_id','action']].copy()
    act_time = pd.get_dummies(act_time, columns=['action'])
    act_time = act_time[['assignment_log_id','timestamp','problem_id','action_correct_response','action_wrong_response','action_problem_finished']]

    # 1 for first response on a problem
    act_time['p_start'] = act_time['action_correct_response'] + act_time['action_wrong_response']

    # keep only needed rows
    act_time = act_time[(act_time['p_start'] ==1)]

    # keep only first response action whether correct or not
    act_time = act_time.sort_values('timestamp').drop_duplicates(subset=['assignment_log_id','problem_id','p_start'])
    
    # LKT format
    act_time['CF..ansbin.'] = act_time['action_correct_response']
    actions = act_time[['assignment_log_id','timestamp','problem_id','CF..ansbin.']].copy()
#%% problem_details

pr_d = actions.merge(data['problem_details'][['problem_id','problem_skill_code']],how='right',on='problem_id')
  
#%% assignment_details
 
ad = pr_d.merge(data['assignment_details'][['assignment_log_id','student_id']].drop_duplicates(),how='left',on='assignment_log_id')

#%% assigment_relationships
 
ar = data['assignment_relationships'].merge(pr_d,left_on='in_unit_assignment_log_id',right_on='assignment_log_id')

#%% finally, add final score

tuts = data['training_unit_test_scores'].merge(data['problem_details'][['problem_id','problem_skill_code']].drop_duplicates(),how='left',on='problem_id')
# euts = data['evaluation_unit_test_scores'].merge(data['problem_details'][['problem_id','problem_skill_code']].drop_duplicates(),how='left',on='problem_id')

# similar to target encoded skill in M2, but that was a mean of problem_skill_code only
tuts_red = tuts.groupby(['unit_test_assignment_log_id','problem_skill_code'])['score'].mean().reset_index().drop_duplicates()
tuts_red.columns = ['unit_test_assignment_log_id', 'problem_skill_code', 'mean_skill_score']
# tuts_score = tuts.merge(tuts_red,on=['unit_test_assignment_log_id','problem_skill_code'])

# yes
tuts_final = tuts_red.merge(ar,how='inner',on=['unit_test_assignment_log_id','problem_skill_code'])
# add back student_id
tuts_final = tuts_final.merge(data['assignment_details'][['assignment_log_id','student_id']].drop_duplicates(),how='inner',on='assignment_log_id')
# sort
tuts_final = tuts_final.sort_values(by=['student_id','timestamp'])

# update outcome var for LKT
mapping = {0: "INCORRECT", 1: "CORRECT"}
if first_attempts: 
    tuts_final['Outcome'] = tuts_final['CF..ansbin.'].replace(mapping)
else: 
    tuts_final['Outcome'] = tuts_final['action_correct_response'].replace(mapping)

# write csv
odir = '/Users/WBR/walter/local_professional/EDM_Cup/'
tuts_final[['student_id','timestamp','Outcome','CF..ansbin.','problem_skill_code','mean_skill_score']].to_csv(odir + 'first-attempts_data_for_LKT.csv',index=False)

#%%


# euts = euts.merge(ar,how='right',on=['unit_test_assignment_log_id','problem_skill_code'])




#%%
 
#%%
 
#%%
 
#%%
 
#%%
 
#%%
 
#%%
 
#%%
 
#%%
 
#%%
  