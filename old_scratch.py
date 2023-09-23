#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:07:15 2023

@author: WBR
"""



#%%
# how to map mean student predicted score to evaluation df? 

# prediction columns
input_cols = ['student_id','mean_prop_correct','unit_test_assignment_log_id','score']
# Collect the input and target columns for the regression
input_cols = [c for c in tuts.columns if 'action' in c]
target_col = 'score'

# Initialize a logistic regression
lr = LogisticRegression(max_iter=1000)
# Fit the regression on all the training data
lr = lr.fit(tuts[input_cols], tuts[target_col])
# Predict the score for each evaluation problem
euts[target_col] = lr.predict_proba(euts[input_cols])[:,1]

# Export the id and score columns of the evaluation unit test scores file for uploading to Kaggle
euts[['id', 'score']].to_csv('/kaggle/working/example_submission.csv', index=False)

#%% Questions not clearly answered in kaggle description

# get mean assignments score for each student from action_logs
# peek = peek_df(action_logs,NROWS=1000)
# looking at action_logs, no straightforward accuracy data
# get some descriptive stats on the actions
# action_sums = action_logs.groupby('action')['timestamp'].count()

# Are the same students in training and evaluation sets?

# What ratio of in_unit_assignment_log_id and unit_test_assignment_log id?
assignment_relationships.in_unit_assignment_log_id.nunique() # n=56577
assignment_relationships.unit_test_assignment_log_id.nunique() #n=638528

# Are the same test problems present in the training and evaluation sets? 
# Get training unit test score unique problems
u_prob = training_unit_test_scores.problem_id.unique()
len(u_prob) # 1835
# Get evaluation unit test score unique problems
u_e_prob = evaluation_unit_test_scores.problem_id.unique()
len(u_e_prob) # 1471 

# check for overlap
set1 = set(u_prob)
set2 = set(u_e_prob)
set3 = set1.intersection(set2)
len(set3) # 1406


# Are assignment log ids unique to each student? 
u_ids =  df.groupby('assignment_log_id')['student_id'].nunique()
all(u_ids == 1) # True

# How to associate training logs with test problems?
# # Associate the action logs for each in unit assignment with their unit test assignment
# df = ar.merge(al, how='left', left_on='in_unit_assignment_log_id', right_on='assignment_log_id')
# df = df[['unit_test_assignment_log_id', 'action']]



#%% Second pass questions

# How many assignments did each student complete? 

# Associate accuracy on set of [which level] problems with correct unit test 
# Break down steps 

#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%


#%%
peek = peek_df(training_unit_test_scores,100)
peek2 = peek_df(euts,100)
peek3 = peek_df(tuts,100)
tuts2.columns
euts.columns
len(tuts)
len(euts)
euts.student_id.nunique()

peek3 = peek_df(tuts[input_cols],100)

tuts2 = tuts[input_cols].copy().drop_duplicates()
tuts.student_id.nunique()
tuts2.student_id.nunique()

tuts2['mean_score'] = tuts2.groupby('student_id')['score'].transform('mean')
tuts2 = tuts2.drop('score',axis=1)
tuts2 = tuts2.drop_duplicates()
tuts2 = tuts2.drop('unit_test_assignment_log_id',axis=1)
tuts2 = tuts2.drop_duplicates()
# 1 more row than n student ids
tuts2['is_duplicate'] = tuts2['student_id'].duplicated()
tuts2['is_duplicate'].sum()
tuts2['nan_student'] = tuts2['student_id'].isna()
tuts2['nan_student'].sum()
tuts2 = tuts2.dropna()
len(tuts2) # 1 na student row dropped

# plot
sns.regplot(tuts2,x='mean_prop_correct',y='mean_score',fit_reg=True)
# oof
#%%