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

# convert timestamp
data['action_logs']['timestamp'] = pd.to_datetime(data['action_logs']['timestamp'], unit='s')

# update column_names
column_names = get_column_names(data)

# action_logs : whittle down to correct, incorrect actions, timestamp, and id info
actions = data['action_logs'][['assignment_log_id','timestamp','problem_id','action']].copy()
actions = actions[(actions['action'] == 'correct_response') | (actions['action'] == 'wrong_response')]
actions = pd.get_dummies(actions,columns=['action'])
actions = actions.drop(columns=['action_wrong_response']) 


#%% optionally whittle actions down to first attempts

first_attempts = True

if first_attempts:
    act_time = data['action_logs'][['assignment_log_id','timestamp','problem_id','action']].copy()
    act_time = pd.get_dummies(act_time, columns=['action'])
    act_time = act_time[['assignment_log_id','timestamp','problem_id','action_correct_response','action_wrong_response','action_problem_finished']]

    # 1 for first response on a problem
    act_time['p_start'] = act_time['action_correct_response'] + act_time['action_wrong_response']

    # keep only needed rows
    act_time = act_time[(act_time['p_start'] ==1)]

    # keep only first response action whether correct or not
    act_time = act_time.sort_values('timestamp').drop_duplicates(subset=['assignment_log_id','problem_id','p_start'],keep='first')
    
    # LKT format
    act_time['CF..ansbin.'] = act_time['action_correct_response']
    actions = act_time[['assignment_log_id','timestamp','problem_id','CF..ansbin.']].copy()
    
else:
    actions['CF..ansbin.'] = actions['action_correct_response']
    actions = actions[['assignment_log_id','timestamp','problem_id','CF..ansbin.']].copy()

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
ar_euts.to_csv(odir + 'd4LKT_10-19.csv',index=False)

#%%