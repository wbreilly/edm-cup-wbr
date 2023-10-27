#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:11:57 2023

@author: WBR
"""


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

# action_logs : whittle down to correct, incorrect actions, timestamp, and id info
actions = data['action_logs'][['assignment_log_id','timestamp','problem_id','action']].copy()
actions['timestamp'] = pd.to_datetime(actions['timestamp'], unit='s')
actions = pd.get_dummies(actions,columns=['action'])

# compute Durations
durations = actions[['assignment_log_id','timestamp','problem_id','action_problem_started','action_problem_finished']].copy()
durations = durations[(durations['action_problem_started'] ==1) | (durations['action_problem_finished'] == 1)]
durations.sort_values('timestamp',inplace=True)
durations['Duration..sec.'] = durations.groupby(['assignment_log_id','problem_id'])['timestamp'].transform('diff')
durations = durations[['assignment_log_id','problem_id','Duration..sec.']].dropna()

# filter actions for responses
actions = actions[(actions['action_correct_response'] == 1) | (actions['action_wrong_response'] == 1)]
actions = actions[['assignment_log_id','problem_id','timestamp','action_correct_response']] 

# add durations to actions
actions = actions.merge(durations,on=['assignment_log_id','problem_id'])

# update column_names
column_names = get_column_names(data)

#%% optionally whittle actions down to first attempts

first_attempts = True

if first_attempts:
   
    # keep only first response action whether correct or not
    actions = actions.sort_values('timestamp').drop_duplicates(subset=['assignment_log_id','problem_id'],keep='first')
    
    # LKT format accuracy
    actions['CF..ansbin.'] = actions['action_correct_response']
    
else:
    actions['CF..ansbin.'] = actions['action_correct_response']

# check that first_attempts isn't messing things up too much 
# nunique_first_attempts = len(actions.drop_duplicates(subset=['assignment_log_id','problem_id']))
# nunique_all_attempts = len(actions.drop_duplicates(subset=['assignment_log_id','problem_id']))
# nunique_first_attempts == nunique_all_attempts #True

#%% problem_details

pr_d = actions.merge(data['problem_details'][['problem_id','problem_skill_code']].drop_duplicates(),how='left',on='problem_id')
  
#%% assignment_details
 
ad = pr_d.merge(data['assignment_details'][['assignment_log_id','student_id']].drop_duplicates(),how='left',on='assignment_log_id')

#%% assigment_relationships
 
ar = ad.merge(data['assignment_relationships'].drop_duplicates(),how='left',right_on='in_unit_assignment_log_id',left_on='assignment_log_id')

#%% filter ar for euts data

# only data that can map to euts
euts = data['evaluation_unit_test_scores'].copy()
euts_ids = euts['unit_test_assignment_log_id'].unique()

# filter
ar_euts = ar[ar['unit_test_assignment_log_id'].isin(euts_ids)].copy()
ar_euts['unit_test_assignment_log_id'].nunique() #10966 

#%% final prep and write csv

# update outcome var for LKT
mapping = {0: "INCORRECT", 1: "CORRECT"}
if first_attempts: 
    ar_euts['Outcome'] = ar_euts['CF..ansbin.'].replace(mapping)
else: 
    ar_euts['Outcome'] = ar_euts['action_correct_response'].replace(mapping)

# fill na psc
ar_euts['problem_skill_code'].fillna('no_psc',inplace=True)

# should be all 0s
pnana = proportion_nans(ar_euts) # checks out

# write csv
odir = '/Users/WBR/walter/local_professional/EDM_Cup/'
ar_euts.to_csv(odir + 'd4LKT_10-24.csv',index=False)

#%%