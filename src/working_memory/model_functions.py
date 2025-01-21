

import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib import gridspec
from time import process_time
import statsmodels.api as sm
import json


import sys
sys.path.append('G:\Mi unidad\WORKING_MEMORY\EXPERIMENTS\ELECTROPHYSIOLOGY\ANALYSIS\functions')
import ephys_functions as ephys
import behavioral_functions as beh


def unnesting(df, explode):
    """
    Unnest columns that contain list creating a new row for each element in the list.
    The number of elements must be the same for all the columns, row by row.
    """
    length = df[explode[0]].str.len()
    idx = df.index.repeat(length)
    df1 = pd.concat([pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx
    finaldf = df1.join(df.drop(explode, 1), how='left')
    finaldf.reset_index(drop=True, inplace=True)

    length2 = [list(range(l)) for l in length]
    length2 = [item + 1 for sublist in length2 for item in sublist]
    name = explode[0] + '_index'
    finaldf[name] = length2

    for column in finaldf.columns:
        try:
            if set(finaldf[column]) <= {'True', 'False', 'nan'}:
                replacing = {'True': True, 'False': False, 'nan': np.nan}
                finaldf[column] = finaldf[column].map(replacing)
        except:
            pass
    return finaldf

def streak(row):
    value = 0
    '''
    Skipping the first trial, look for whether there was a sequences of same side trials or opposite side trials
    '''
    if row['trial'] != 1:
        if row['stim'] == row['past_choices'][0]:
            value = 1
            if row['past_choices'][0] == row['past_choices'][1]:
                value = 2
                if row['past_choices'][1] == row['past_choices'][2]:
                    value = 3
        else: 
            value = -1
            if row['past_choices'][0] == row['past_choices'][1]:
                value = -2
                if row['past_choices'][1] == row['past_choices'][2]:
                    value = -3 
                else:
                    value = -2
                    return value
            else:
                value = -1
                return value
    return value

# Create a repeating vector
def repeat_choice(row):
    val = 0
    if row['choices'] == row['past_choices'][0]:
        val = 1
    elif row['past_choices'][0] == 0:
        val = np.nan
    else:
        val = 0
    return val

def repeat_choice_side(row):
    '''
    Parameters
    ----------
    row : look row by row whether there was a repetition of LEFT response (1) or RIGHT response (2)

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # Compare the current response with the previous one. If that matches, return a 1 meaning it repeated. 
    if row['trial'] != 0:
        # Compare the current response with the previous one. .  
        if row['choices'] == row['past_choices'][0]:
            # if it matched and the answer was 1, it means that it repeated a right response
            if row['choices'] == 1:
                return 2
            # if it matched and the answer was 0, it means that it repeated a left response
            else:
                return 1
        elif row['past_choices'][0] == 0:
            return 0
        else:
            return 0
    else:
        return np.nan
    
def fix_delays_model(row, set_delay = 1):
    if set_delay == 0:
        delays=[0.0,500,1000,2000,3000,4000,5000,7000,9000,10000]
    elif set_delay == 1:
        delays=[100,1000,3000,10000]
    return delays[row['idelays']-1]/1000

def fix_delays(row):
    if set_delay == 0:
        delays=[0.0,500,1000,2000,3000,4000,5000,7000,9000,10000]
    elif set_delay == 1:
        delays=[100,1000,3000,10000]
    return delays[row['idelays']-1]/1000

def create_data(animal,extra):
    os.getcwd() 
    os.chdir('C:/Users/Tiffany/Desktop/HMM_mice/real/')

    with open(animal+'_behavior_'+extra+'.json', 'r') as j:
        json_data = json.load(j)

    df = pd.DataFrame.from_dict(json_data, orient='columns', dtype=None, columns=None)

    with open(animal+'_fit_'+extra+'.json', 'r') as j:
        json_data = json.load(j)

    fit = pd.DataFrame.from_dict(json_data, orient='columns', dtype=None, columns=None)
    fit.ParamFit = np.around(fit.ParamFit,3)
    
    # Reset index and create column for session
    df.reset_index(inplace=True)
    df.rename(columns = {'index':'session'}, inplace = True)

    # Unnest several parameters so it is one per row
    new_df = unnesting(df, ['choices', 'stim','idelays','day'])

    # Prepare the past choices
    addition=[]
    for k in range(len(df.past_choices)):
        for i in range(len(df['past_choices'][k])):
            addition.append(df['past_choices'][k][i])
    new_df['past_choices'] = addition

    # Prepare past rewards
    addition=[]
    for k in range(len(df.past_rewards)):
        for i in range(len(df['past_rewards'][k])):
            addition.append(df['past_rewards'][k][i])
    new_df['past_rewards'] = addition

    # Prepare states
    addition=[]
    for k in range(len(df.Pstate)):
        for i in range(len(df['Pstate'][k])):
            addition.append(df['Pstate'][k][i])
    new_df['state'] = addition

    # ---------------
    def after_correct(row):
        return row['past_rewards'][0]

    new_df['after_correct'] = new_df.apply(after_correct, axis=1)
    
    # Next delays
    new_df['delays'] = new_df.apply(fix_delays_model, axis=1)

    # -------
    new_df.rename(columns = {'choices_index':'trial'}, inplace = True)
    new_df['session'] = new_df['session'].astype(int)

    # Revalue variables stim, choices and create hit vector. -1 means Left and 1 means right. 1 for correct in hit and 0 for incorrect
    new_df['stim']= np.where(new_df['stim'] == 1, -1, 1)
    new_df['choices']= np.where(new_df['choices'] == 1, -1, 1)
    new_df['hit'] = np.where(new_df['stim'] == new_df['choices'], 1, 0)

    new_df['repeat'] = new_df.apply(repeat_choice,axis=1)

    new_df['repeat_choice_side'] = new_df.apply(repeat_choice_side,axis=1)

    temp = pd.DataFrame(new_df['state'].tolist())
    new_df['WM'] = temp[0]
    new_df['RL'] = temp[1]
    
    # Correct the sequences of repetitions and alternations
    new_df['streak'] = new_df.apply(streak,axis=1)

    new_df.drop(columns = {"state", "idelays","Pstate"}, inplace=True)
    
    return new_df, fit