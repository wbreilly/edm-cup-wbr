#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:01:31 2023

@author: WBR
"""

# M7 script is a bit more elaborate
# First, it prepares a csv for R. This csv contains the action lgos for the 
# first attempts of practice problems completed in assigments that are linked
# to the evaluation test set
# Next, in R, I ran the data through a Recency Performance Factors Analysis 
# using the LKT Package
# Back on this script, I used the model final practice problem prediction to
# make the test 'score' predictions
# Additionally, I investigated why a large proportion of 'score' rows were nan,
# concluding that a feature of the dataset is that there are many test problems
# for which there is no matching student and problem skill code practice 
# Bonus: submission 'score need not be binary. Boost in accuracy from continuous 


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
########################################################################################
#%% After running LKT model in R, merge predictions with euts based on id and psc

odir = '/Users/WBR/walter/local_professional/EDM_Cup/'
# model6_euts = pd.read_csv(odir + "model6_euts_d4LKT.csv") # made with d4LKT.csv, 45 % na
model6_euts = pd.read_csv(odir + "model6_euts_d4LKT_10-19.csv")

# make sure no one got lost in R...
model6_euts['unit_test_assignment_log_id'].nunique() # 10966

# use prediction on last trial of problem_skill_code
model6_euts = model6_euts.sort_values(by=['unit_test_assignment_log_id','timestamp'])
model6_euts_reduce = model6_euts.drop_duplicates(subset=['unit_test_assignment_log_id','problem_skill_code'],keep='last').copy()
# model6_euts_reduce = model6_euts.copy()

# add psc
euts = data['evaluation_unit_test_scores'].merge(data['problem_details'][['problem_id','problem_skill_code']].drop_duplicates(),how='left',on='problem_id')
euts.drop('score',inplace=True,axis=1)
# fill nas in psc
euts['problem_skill_code'].fillna('no_psc',inplace=True)

# threshold model predictions

# mean training score. Mean predicted score should be similar.
# data['training_unit_test_scores'].score.mean() #.585
 
# parameter sweep  of thresholds until mean of evaluation score is near mean of training score
for thresh in np.arange(.7,.8,step=.01):
    tempmean = np.where(model6_euts_reduce['pred'] < thresh,1,0).mean()
    print('thresh: ', thresh,'\n mean: ',tempmean)
# .78
model6_euts_reduce['score'] = np.where(model6_euts_reduce['pred'] > .78,1,0)

# 
check = model6_euts_reduce[['unit_test_assignment_log_id','problem_skill_code','score']].drop_duplicates() 
check = check.merge(euts,how='right', on=['unit_test_assignment_log_id','problem_skill_code']) 

pnana = proportion_nans(check) # 45 %

#%% Fill na's based on mean performance

group_mean = check.groupby('unit_test_assignment_log_id')['score'].transform('mean')
check['score'] = check['score'].fillna(group_mean)
pnana = proportion_nans(check) # still have 10% na 

problem_mean = check.groupby('problem_id')['score'].transform('mean')
check['score'] = check['score'].fillna(problem_mean)
pnana = proportion_nans(check) # still have .003 nans

check['score'].fillna(check['score'].mean(),inplace=True)

check[['id', 'score']].to_csv('/Users/WBR/walter/local_professional/EDM_Cup/submissions/' + 'm7.csv', index=False)

# threshold 'score'
check['score'] = np.where(check['score'] > .5,1,0)
check[['id', 'score']].to_csv('/Users/WBR/walter/local_professional/EDM_Cup/submissions/' + 'm7-2.csv', index=False)
# This decreased private and public scores. Surprising because I thought only binary values would be  considered

# What happens if I don't threshold 
nothresh = model6_euts_reduce[['unit_test_assignment_log_id','problem_skill_code','pred']].drop_duplicates() 
nothresh = nothresh.merge(euts,how='right', on=['unit_test_assignment_log_id','problem_skill_code']) 
nothresh['score'] = nothresh['pred']
group_mean = nothresh.groupby('unit_test_assignment_log_id')['score'].transform('mean')
nothresh['score'] = nothresh['score'].fillna(group_mean)
problem_mean = nothresh.groupby('problem_id')['score'].transform('mean')
nothresh['score'] = nothresh['score'].fillna(problem_mean)
nothresh['score'].fillna(nothresh['score'].mean(),inplace=True)
pnana = proportion_nans(nothresh)

nothresh[['id', 'score']].to_csv('/Users/WBR/walter/local_professional/EDM_Cup/submissions/' + 'm7-3.csv', index=False)
# this produced highest private score i've seen! from .61 previous to .65
#%%
#######################################
#%% Investigate why 45% of 'score' are na

# start with clean euts + psc  
euts = data['evaluation_unit_test_scores'].merge(data['problem_details'][['problem_id','problem_skill_code']].drop_duplicates(),how='left',on='problem_id')

# some baseline comparisons
euts_s_psc = euts[['unit_test_assignment_log_id','problem_skill_code']].drop_duplicates() #40123
m6_s_psc = model6_euts[['unit_test_assignment_log_id','problem_skill_code']].drop_duplicates() #50720
 
euts_s_psc.problem_skill_code.nunique() #309
m6_s_psc.problem_skill_code.nunique() # 447

euts_s_psc.problem_skill_code.isin(m6_s_psc.problem_skill_code.unique()).sum() #38228

# filter dataframe to compare instead of dictionary method
m6_filt = m6_s_psc[m6_s_psc['problem_skill_code'].isin(euts_s_psc.problem_skill_code.unique())]
m6_filt.problem_skill_code.nunique() #302
m6_filt['unit_test_assignment_log_id'].nunique() #10591

m6_filt_s =  m6_filt[m6_filt['unit_test_assignment_log_id'].isin(euts_s_psc.unit_test_assignment_log_id.unique())] # no change

# # old
# # only 7 pscs missing in model from euts. Good! 
# # 166 students missing after filtering for euts pscs
# # how many rows in euts would 166 students take up?
# euts.groupby('student_id')['id'].count().mean() # 16 rows per student
# 166 * 16 / 124455 # 2 % of rows. Doesn't explain missing score data of 50%

#%% use sets to compare unique combinations of ids and pscs in euts and action logs used in m6

# make dicts
m6_filt_s_dict = m6_s_psc.groupby('unit_test_assignment_log_id')['problem_skill_code'].apply(list).to_dict()
euts_s_psc_dict = euts_s_psc.groupby('unit_test_assignment_log_id')['problem_skill_code'].apply(list).to_dict()

# make sets
m6_set = {key: set(value) for key, value in m6_filt_s_dict.items()}
euts_set = {key: set(value) for key, value in euts_s_psc_dict.items()}

# find missing id/psc pairs
euts_missing_values = {}
missing_keys = {}
# Iterate through the keys of the dictionaries
for key in euts_set.keys():
    if key in m6_set.keys():
        missing = list(euts_set[key] - m6_set[key])
    else:
        missing_keys[key] = None
    if missing:
        euts_missing_values[key] = missing
# 8310 test ids are missing problem_skill_codes from practice  

# the total number of values in euts_missing_values gives part of the picture of missing scores. 
# Because many test problems that have the same psc
total_missing_count = 0
# Iterate through the dictionary and count missing values for each key
for key, values in euts_missing_values.items():
    missing_count = len(values)
    total_missing_count += missing_count
# 18122

# filter rows based on index of missing pscs - final step to pin down this issue as cause of missing scores
# these are 
df_missing = pd.DataFrame([(key, value) for key, values in euts_missing_values.items() for value in values], columns=['unit_test_assignment_log_id', 'problem_skill_code'])
df_missing = df_missing.merge(check, on=['unit_test_assignment_log_id','problem_skill_code']) # 30% ofmissing EUTS rows accounted for

len(df_missing) / len(euts) # 32 % missing accounted for 

pnana = proportion_nans(df_missing)

#%% finally, look at nans that are not accounted for by df_missing

# Merge the DataFrames with indicator=True
merged = pd.merge(df_missing, check, on=['unit_test_assignment_log_id','problem_skill_code'], how='outer', indicator=True)
# Filter rows that are only in the second DataFrame
inverse_merged = merged[merged['_merge'] == 'right_only']

len(check) - len(df_missing) == len(inverse_merged) # True

pnana = proportion_nans(inverse_merged)
# still have score nans even though psc is not nan

look = check[check['problem_skill_code'] == 'no_psc']
look2 = df_missing[df_missing['problem_skill_code'] == 'no_psc'] # 0 

(len(look) + len(df_missing)) / len(check) # .448 vs .445 in check
pnana = proportion_nans(look) # 83% are nan for score, not 100

(len(look)*.83 + len(df_missing)) / len(check) #.425

#%%

# In conclusion, we can account for 42.5% of the nan's for 'score' out of the 
# total 45%.

# 32% are accounted for due to a large number of unit_test_asignment_log_ids 
# lacking practice data for a tested problem_skill_code.

# A further 10% are accounted for due to there being no psc  for the test 
# problem. 

# Clearly, the most accurate model would predict performance for a given 
# student on a given skill based on that student's practice performance on 
# that same skill.

# These data were likely removed from the dataset to require model transfer to
# unseen students.

# Therefore, proceeding models will consider these constraints.  


#%%






































