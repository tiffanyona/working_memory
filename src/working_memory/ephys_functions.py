
import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib import gridspec
from time import process_time
import statsmodels.api as sm
import seaborn as sns

from sklearn.utils import shuffle
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import sys
sys.path.append('G:\Mi unidad\WORKING_MEMORY\EXPERIMENTS\ELECTROPHYSIOLOGY\ANALYSIS\functions')
import model_functions as mod
import behavioral_functions as beh

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
# warnings.simplefilter(action='ignore', category=PerformanceWarning)
warnings. filterwarnings('ignore', category=UserWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None  # default='warn'

def plot_results_decoder(fig, real, times, df_new,  plot, color = 'black', epoch = 'Delay_OFF', 
                         y_range = [-0.05, 0.3], x_range = None, substract=True):
    if x_range == None:
        x_range = [min(times), max(times)]

    shuffle_mean = df_new.mean().values

    if substract == True:
        lower =  df_new.quantile(q=0.975, interpolation='linear')-shuffle_mean
        upper =  df_new.quantile(q=0.025, interpolation='linear')-shuffle_mean
        corrected_real = real-shuffle_mean
    else:
        lower =  df_new.quantile(q=0.975, interpolation='linear')
        upper =  df_new.quantile(q=0.025, interpolation='linear')
        corrected_real = real    

    plot.plot(times,corrected_real, color=color)
    plot.plot(times, lower+corrected_real, color=color, linestyle = '',alpha=0.6)
    plot.plot(times, upper+corrected_real, color=color, linestyle = '',alpha=0.6)
    # plot.fill_between(times, lower+corrected_real, upper+corrected_real, alpha=0.2, color=color)
    plot.axhline(y=0.0,linestyle=':',color='black')
    plot.set_ylim(y_range)
    plot.set_xlim(x_range)

    if epoch == 'Stimulus_ON':
        plot.set_xlabel('Time to stimulus onset (s)')
        plot.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.4, color='grey', alpha=.4)
    else:
        plot.set_xlabel('Time to go cue (s)')
        plot.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.2, color='grey', alpha=.4)

def plot_results_session_summary(plot, df, colors, variables_combined = ['WM_roll_1', 'RL_roll_1'], 
                                 y_range = [], x_range = None, epoch = 'Stimulus_ON', baseline=0.5):
    
    for color, variable, ax in zip(colors, variables_combined, np.repeat(plot, len(variables_combined))):
        try:
            df_loop = df.loc[(df['trial_type'] == variable)]
        except:
            df_loop = df

        df_loop = df_loop.dropna(axis=1, how='all')

        # Select only columns where the column name is a number or can be transformed to a number
        numeric_columns = df_loop.columns[df_loop.columns.to_series().apply(pd.to_numeric, errors='coerce').notna()]

        real = np.mean(df_loop.groupby('session')[numeric_columns].mean(), axis=0).to_numpy()
        times = df_loop[numeric_columns].columns.astype(float)


        df_results = pd.DataFrame()
        df_results['times'] = times
        df_results['real'] = real
        df_results = df_results.sort_values(by='times')
        

        if x_range == None:
            x_range = [min(times), max(times)]
            
        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        df_for_boots = df_loop.groupby('session')
        for timepoint in df_results['times'].values:
            mean_surr = []

            # recover the values for that specific timepoint
            try:
                array = df_for_boots[timepoint].mean().to_numpy()
            except:
                array = df_for_boots[str(timepoint)].mean().to_numpy()

            # iterate several times with resampling: chose X time among the same list of values
            for iteration in range(1000):
                x = np.random.choice(array, size=len(array), replace=True)
                # recover the mean of that new distribution
                mean_surr.append(np.mean(x))

            df_lower.at[0, timepoint] = np.percentile(mean_surr, 2.5)
            df_upper.at[0, timepoint] = np.percentile(mean_surr, 97.5)

        lower =  df_lower.iloc[0].values
        upper =  df_upper.iloc[0].values

        ax.plot(df_results.times,df_results.real, color=color)
        ax.fill_between(df_results.times, lower, upper, alpha=0.2, color=color)
        ax.axhline(y=baseline,linestyle=':',color='black')
        ax.set_ylim(y_range)
        ax.set_xlim(x_range)

        if epoch == 'Stimulus_ON':
            ax.set_xlabel('Time to stimulus onset (s)')
            ax.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.4, color='grey', alpha=.4)
        else:
            ax.set_xlabel('Time to go cue (s)')
            ax.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.2, color='grey', alpha=.4)

        sns.despine()

def plot_results_session_summary_substract(fig, plot, df: pd.DataFrame, df_shuffle: pd.DataFrame, color, variable= 'WM_roll_1', 
                                 y_range = [-0.05, 0.3], x_range = None, epoch = 'Stimulus_ON', baseline=0.5):
    
        df_loop = (df.loc[(df['trial_type'] == variable)].groupby('session').mean()
                    - df_shuffle.loc[(df['trial_type'] == variable)].groupby('session').mean())

        # Select only columns where the column name is a number or can be transformed to a number
        numeric_columns = df_loop.columns[df_loop.columns.to_series().apply(pd.to_numeric, errors='coerce').notna()]

        real = np.array(np.mean(df_loop.groupby('session').mean()[numeric_columns])) 
        times = np.array(np.mean(df_loop[numeric_columns]).index).astype(float)

        df_results = pd.DataFrame()
        df_results['times'] = times
        df_results['real'] = real
        df_results = df_results.sort_values(by='times')

        if x_range == None:
            x_range = [min(times), max(times)]
            
        mean_surr = []
        df_lower = pd.DataFrame()
        df_upper = pd.DataFrame()

        df_for_boots = df_loop.groupby('session').mean()
        for timepoint in times:
            mean_surr = []

            # recover the values for that specific timepoint
            try:
                array = df_for_boots[timepoint].to_numpy()
            except:
                array = df_for_boots[str(timepoint)].to_numpy()

            # iterate several times with resampling: chose X time among the same list of values
            for iteration in range(1000):
                x = np.random.choice(array, size=len(array), replace=True)
                # recover the mean of that new distribution
                mean_surr.append(np.mean(x))

            df_lower.at[0, timepoint] = np.percentile(mean_surr, 2.5)
            df_upper.at[0, timepoint] = np.percentile(mean_surr, 97.5)

        lower =  df_lower.iloc[0].values
        upper =  df_upper.iloc[0].values

        plot.plot(df_results.times,df_results.real, color=color)
        plot.fill_between(df_results.times, lower, upper, alpha=0.2, color=color)
        plot.axhline(y=baseline,linestyle=':',color='black')
        plot.set_ylim(y_range)
        plot.set_xlim(x_range)

        if epoch == 'Stimulus_ON':
            plot.set_xlabel('Time to stimulus onset (s)')
            plot.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.4, color='grey', alpha=.4)
        else:
            plot.set_xlabel('Time to go cue (s)')
            plot.fill_betweenx(np.arange(-1,1.15,0.1), 0,0.2, color='grey', alpha=.4)

        sns.despine()

def add_to_summary(real, shuffle_mean, times, filename, variable_full, epoch, fold_no, df_iter, 
                   df_cum, df_cum_shuffle, score = np.nan, substract=True, delay:float=-1):

    if substract == True:
        a_series = pd.DataFrame(pd.Series(real-shuffle_mean, index = times)).T
    else:
        a_series = pd.DataFrame(pd.Series(real, index = times)).T
        
    a_series['trial_type'] = variable_full
    a_series['session'] = filename
    a_series['epoch'] = epoch
    a_series['score'] = score    
    a_series['fold'] = fold_no 
    if delay > 0:
        a_series['delay'] = delay
        
    # df_cum = df_cum.append(a_series, ignore_index=True)
    df_cum = pd.concat([df_cum, a_series], ignore_index=True)
    
    df_cum_iter=pd.DataFrame()
    df_cum_iter['times'] = df_iter.groupby('times').score.mean().reset_index()['times'].values
    df_cum_iter['epoch'] = epoch
    df_cum_iter['session'] = filename
    df_cum_iter['fold'] = fold_no 
    df_cum_iter['trial_type'] = variable_full 
    if delay > 0:
        df_cum_iter['delay'] = delay
    if substract == True:
        for iteration in df_iter.iteration.unique():
            df_cum_iter[iteration]= df_iter.loc[(df_iter.variable==variable_full)&(df_iter.iteration==iteration)&(df_iter.epoch==epoch)].groupby('times')['score'].mean().values - df_iter.groupby('times')['score'].mean().values
    else:
        for iteration in df_iter.iteration.unique():
            df_cum_iter[iteration]= df_iter.loc[(df_iter.variable==variable_full)&(df_iter.iteration==iteration)&(df_iter.epoch==epoch)].groupby('times')['score'].mean().values

    df_cum_shuffle = pd.concat([df_cum_iter, df_cum_shuffle])

    return df_cum, df_cum_shuffle

def train_test_shuffle(df, decode='vector_answer', epoch='Delay_OFF', cluster_list=[],
               test_index=[],  train_index=[], initrange=-0.4, endrange=1.5, r=0.2, delay_only=False,
               variable_test='WM_roll', hit_test=1, ratio_test=0.6, nsurrogates=10, 
               variable_train='all', ratio_train=0.5, hit_train=1, score_options = 'standard', reduce_WM = 'all'):

    df_real = pd.DataFrame()
    df_iter = pd.DataFrame(columns = ['iteration','score', 'times','epoch' ,'variable'])
    index_iter = 0

    times = []  # Timestamps
    real_score = []  # real scoring of the decoded

    for start, stop in zip(np.arange(initrange, endrange-r, r), np.arange(initrange+r, endrange, r)):
        # print((start+stop)/2)
        if hit_train == 'all' and variable_train == 'all':
            df_train = df
        elif hit_train == 'all':
            df_train = df.loc[(df['state']  == variable_train[:2])]
        elif variable_train == 'all':
            df_train = df.loc[(df['hit'] == hit_train)]
        else:
            df_train = df.loc[(df['state']  == variable_train[:2])
                              & (df.hit == hit_train)]

        df_final, y = interval_extraction(df_train, decode=decode, align=epoch, start=start, stop=stop,
                                          cluster_list=cluster_list, delay_only=delay_only)

        train_cols = df_final.columns
        trained_trials = df_final.iloc[train_index, :].index.values

        # Train the model
        df_final = df_final.reset_index()
        df_final = df_final.drop(columns='trial')
        x = df_final.iloc[:, df_final.columns != 'y']

        try:
            if len(test_index) >= 1:
                train = df_final.iloc[train_index, :]
                test = df_final.iloc[test_index, :]
                x_test = test.iloc[:, test.columns != 'y']
                y_test = test['y']
                x_train = train.iloc[:, train.columns != 'y']
                y_train = train['y']

            else:
                x_test = df_final.iloc[:, df_final.columns != 'y']
                y_test = df_final['y']
        except:
            print('Error in indexing training trials', (start+stop)/2)

        # Normalize the X data
        # sc = RobustScaler()
        # sc_fit = sc.fit(x)
        # x_train = sc_fit.transform(x_train)

        if hit_test == 'all' and variable_test == 'all':
            df_test = df
        elif hit_test == 'all':
            df_test = df.loc[(df['state'] == variable_test[:2])]
        elif variable_test == 'all':
            df_test = df.loc[(df['hit'] == hit_test)]
        else:
            df_test = df.loc[(df['state'] == variable_test[:2]) & (df.hit == hit_test)]

    # -----------  Remove the trials that overlap with the training set.

        df_test = df_test[~df_test['trial'].isin(trained_trials)]
        # print(len(df_test.trial.unique()))

        count_RL = len(df.loc[(df['state'] == 'RL')]['trial'].unique())

        if count_RL > 8 and variable_test == 'WM_roll' and reduce_WM != False:
            if reduce_WM == 'correct':
                count_RL = len(df.loc[(df['state'] == 'RL')&(df.hit==1)]['trial'].unique())
            elif reduce_WM == 'all': 
                count_RL = len(df.loc[(df['state'] == 'RL')]['trial'].unique())
            else:
                count_RL = 0

            try:
                test_index = np.random.choice(test_index, size=count_RL, replace=False)
            except:
                test_index = np.random.choice(test_index, size=count_RL, replace=True)
        elif count_RL < 8:
            continue

        if len(df_test.trial.unique()) <= 4 or len(df_test.vector_answer.unique()) != 2:
            print('Not enough RL trials - within loop ' + str(len(df_test.trial.unique())))
            # real_score.append(0)
            break


        times.append((start+stop)/2)

        test, y = interval_extraction(df_test, decode=decode, align=epoch, start=start, stop=stop,
                                      cluster_list=cluster_list, delay_only=delay_only)

        x_test = test.iloc[:, test.columns != 'y']
        y_test = test['y']

        # x_test = sc.transform(x_test)

        model = LogisticRegression(solver='liblinear', penalty='l2', C=.99).fit(x_train, y_train)
        train_cols = df_final.columns

        p_pred = model.predict_proba(x_test)
        y_pred = model.predict(x_test)
        
        if score_options == 'standard':
            score_ = model.score(x_test, y_test)

        else:
            y_test = np.where(y_test == -1, 0, y_test)
            y_new = y_test.reshape(len(y_test), 1).astype(int)
            # y_new = y_test.values.reshape(len(y_test), 1).astype(int)
            score_ =  np.take_along_axis(p_pred,y_new,axis=1)

        real_score.append(score_)

        for i in np.arange(nsurrogates):
            y_perr = shuffle(y_test)
           
            if score_options == 'standard':
                score_ = model.score(x_test, y_perr)   
                score_  = np.mean(score_)
            else:
                # y_new = y_perr.values.reshape(len(y_perr), 1).astype(int)
                y_new = y_perr.reshape(len(y_perr), 1).astype(int)
                result =  np.take_along_axis(p_pred,y_new,axis=1)
                score_  = np.mean(result)
                        
            new_row = {'iteration': i, 'score': score_, 'times': (start+stop)/2, 'epoch' : epoch, 'variable' : (str(variable_test)+'_'+str(hit_test))}
        
            
            # Use the `loc` indexer to insert the row
            df_iter.loc[index_iter] = new_row
            index_iter +=1 

    times.append('trial_type')

    real_score.append(variable_test+'_'+str(hit_test))
    a_series = pd.Series(real_score, index=times)
    df_real = df_real.append(a_series, ignore_index=True)

    return df_real, df_iter

def train_test(df, decode='vector_answer', epoch='Delay_OFF', cluster_list=[],
               test_index=[],  train_index=[], initrange=-0.4, endrange=1.5, r=0.2, delay_only=False,
               variable='WM_roll', hit=1, ratio=0.6, nsurrogates=10, 
               variable_train='all', ratio_train=0.5, hit_train=1, score_options = 'standard', reduce_WM = 'all'):

    df_real = pd.DataFrame()
    df_iter = pd.DataFrame(columns = ['iteration','score', 'times','epoch' ,'variable'])
    index_iter = 0

    times = []  # Timestamps
    real_score = []  # real scoring of the decoded

    for start, stop in zip(np.arange(initrange, endrange-r, r), np.arange(initrange+r, endrange, r)):
        # print((start+stop)/2)
        if hit_train == 'all' and variable_train == 'all':
            df_train = df
        elif hit_train == 'all':
            df_train = df.loc[(df[variable_train] > ratio_train)]
        elif variable_train == 'all':
            df_train = df.loc[(df['hit'] == hit_train)]
        else:
            df_train = df.loc[(df[variable_train] > ratio_train)
                              & (df.hit == hit_train)]

        df_final, y = interval_extraction(df_train, decode=decode, align=epoch, start=start, stop=stop,
                                          cluster_list=cluster_list, delay_only=delay_only)

        train_cols = df_final.columns
        trained_trials = df_final.iloc[train_index, :].index.values

        # Train the model
        df_final = df_final.reset_index()
        df_final = df_final.drop(columns='trial')
        x = df_final.iloc[:, df_final.columns != 'y']

        try:
            if len(test_index) >= 1:
                train = df_final.iloc[train_index, :]
                test = df_final.iloc[test_index, :]
                x_test = test.iloc[:, test.columns != 'y']
                y_test = test['y']
                x_train = train.iloc[:, train.columns != 'y']
                y_train = train['y']

            else:
                x_test = df_final.iloc[:, df_final.columns != 'y']
                y_test = df_final['y']
        except:
            print('Error in indexing training trials', (start+stop)/2)

        # Normalize the X data
        # sc = RobustScaler()
        # sc_fit = sc.fit(x)
        # x_train = sc_fit.transform(x_train)

        if hit == 'all' and variable == 'all':
            df_test = df
        elif hit == 'all':
            df_test = df.loc[(df[variable] > ratio)]
        elif variable == 'all':
            df_test = df.loc[(df['hit'] == hit)]
        else:
            df_test = df.loc[(df[variable] > ratio) & (df.hit == hit)]

    # -----------  Remove the trials that overlap with the training set.

        df_test = df_test[~df_test['trial'].isin(trained_trials)]
        # print(len(df_test.trial.unique()))

        count_RL = len(df.loc[(df['RL_roll'] >0.4)]['trial'].unique())

        if count_RL > 4 and variable == 'WM_roll' and reduce_WM != False:
            if reduce_WM == 'correct':
                count_RL = len(df.loc[(df['RL_roll'] >0.4)&(df.hit==1)]['trial'].unique())
            elif reduce_WM == 'all': 
                count_RL = len(df.loc[(df['RL_roll'] >0.4)]['trial'].unique())
            else:
                count_RL = 0

            try:
                test_index = np.random.choice(test_index, size=count_RL, replace=False)
            except:
                test_index = np.random.choice(test_index, size=count_RL, replace=True)
        elif count_RL < 4:
            continue

        if len(df_test.trial.unique()) < 4 or len(df_test.vector_answer.unique()) != 2:
            print('Not enough RL trials - within loop ' + str(len(df_test.trial.unique())))
            # real_score.append(0)
            break


        times.append((start+stop)/2)

        test, y = interval_extraction(df_test, decode=decode, align=epoch, start=start, stop=stop,
                                      cluster_list=cluster_list, delay_only=delay_only)

        x_test = test.iloc[:, test.columns != 'y']
        y_test = test['y']

        # x_test = sc.transform(x_test)

        model = LogisticRegression(solver='liblinear', penalty='l2', C=.99, class_weight='balanced').fit(x_train, y_train)
        train_cols = df_final.columns

        p_pred = model.predict_proba(x_test)
        y_pred = model.predict(x_test)
        
        if score_options == 'standard':
            score_ = model.score(x_test, y_test)

        else:
            y_test = np.where(y_test == -1, 0, y_test)
            y_new = y_test.reshape(len(y_test), 1).astype(int)
            # y_new = y_test.values.reshape(len(y_test), 1).astype(int)
            score_ =  np.take_along_axis(p_pred,y_new,axis=1)

        real_score.append(score_)

        for i in np.arange(nsurrogates):
            y_perr = shuffle(y_test)
           
            if score_options == 'standard':
                score_ = model.score(x_test, y_perr)   
                score_  = np.mean(score_)
            else:
                # y_new = y_perr.values.reshape(len(y_perr), 1).astype(int)
                y_new = y_perr.reshape(len(y_perr), 1).astype(int)
                result =  np.take_along_axis(p_pred,y_new,axis=1)
                score_  = np.mean(result)
                        
            new_row = {'iteration': i, 'score': score_, 'times': (start+stop)/2, 'epoch' : epoch, 'variable' : str(variable)+'_'+str(hit)}
            
            # Use the `loc` indexer to insert the row
            df_iter.loc[index_iter] = new_row
            index_iter +=1 

    times.append('trial_type')

    real_score.append(variable+'_'+str(hit))
    a_series = pd.Series(real_score, index=times)
    df_real = df_real.append(a_series, ignore_index=True)

    return df_real, df_iter
    
def add_time_before_stimulus(df, time_added):
    # ----------------- Add 2 more seconds of the previous trials before the current stimulus
    # df = df.rename(columns={'past_choices_x' : 'past_choices', 'streak_x' : 'streak', 'past_rewards_x' : 'past_rewards'})
    # df = df.drop(columns=['past_choices_y','streak_y', 'past_rewards_y'])

    # Create a DataFrame only with info for the session
    try:
        trials = df.groupby('trial')[['START','END','Delay_ON','Delay_OFF', 'Stimulus_ON', 'Response_ON', 'Lick_ON', 'Motor_OUT','new_trial',
               'vector_answer', 'reward_side', 'hit', 'delay','total_trials', 'T', 'previous_vector_answer', 'previous_reward_side','repeat_choice',
                'WM_roll', 'RL_roll', 'WM', 'RL', 'streak']].mean()
    except:
        trials =  df.groupby('trial')[['START','END','Delay_ON','Delay_OFF', 'Stimulus_ON', 'Response_ON', 'Lick_ON', 'Motor_OUT','new_trial',
               'vector_answer', 'reward_side', 'hit', 'delay','total_trials', 'T', 'previous_vector_answer', 'previous_reward_side','repeat_choice',
                'WM_roll', 'RL_roll', 'WM', 'RL']].mean()
        
    trials = trials.reset_index()

    # Make an aligment to END column
    df['a_END'] = df['fixed_times'] - df['END']

    # Create a new DataFrame with all spikes
    try:
        # Some sessions include the group column that indicates the type of cluste,r other don't
        spikes = df[['trial', 'fixed_times', 'a_END', 'cluster_id', 'group']]
    except:
        spikes = df[['trial', 'fixed_times', 'a_END', 'cluster_id']]

    # Locate spikes that happen 2s prior to end of trial and copy them changing the new_trial index
    duplicate_spikes = spikes.loc[spikes.a_END > + time_added]
    duplicate_spikes['trial'] += 1

    # Add the duplicates
    spikes = pd.concat([spikes, duplicate_spikes])

    # Merge trial data with spikes on trial idnex
    df = pd.DataFrame()
    df = pd.merge(trials, spikes, on=["trial"])

    # Create the columns for start and end and change trial to new trial index ( without taking the misses into account)
    # df['trial_start'] = min(df.new_trial)
    # df['trial_end'] = max(df.new_trial)
    # df = df.drop(columns=['trial'])
    # df = df.rename(columns={'new_trial' : 'trial'})

    # This in case we don't do this and want to preserve the orginal trial indexes.
    df['trial_start'] = min(df.trial)
    df['trial_end'] = max(df.trial)

    # Crate the aligment that ew will need for the analysis.
    df['a_Stimulus_ON'] = df['fixed_times'] - df['Stimulus_ON']
    df['a_Lick_ON'] = df['fixed_times'] - df['Lick_ON']
    df['a_Delay_OFF'] = df['fixed_times'] - df['Delay_OFF']
    df['a_Motor_OUT'] = df['fixed_times'] - df['Motor_OUT']
    df['a_Response_ON'] = df['fixed_times'] - df['Response_ON']
    df['START_adjusted'] = df['START'] + time_added - 0.1

    return df

def interval_extraction(df, cluster_list=[], decode='vector_answer', align='Delay_OFF', start:float=0.0, stop=1.0, delay_only=False):
    y = []
    d = {}

    if delay_only == False:
        # print('Skipping delays')
        if align == 'Delay_OFF' and start < 0:
            df = df.loc[(df.delay != 0.1) & (df.delay != 0.2)]
        if align == 'Delay_OFF' and start < -1.1:
            df = df.loc[(df.delay != 0.1) & (
                df.delay != 0.2) & (df.delay != 1)]

        if align == 'Stimulus_ON' and stop > 0.5:
            df = df.loc[(df.delay != 0.1) & (df.delay != 0.2)]

        if align == 'Stimulus_ON' and stop > 1.5:
            df = df.loc[(df.delay != 0.1) & (
                df.delay != 0.2) & (df.delay != 1)]

    # print('Recovered from: ', str(len(df.trial.unique())), ' trials')
    # Create new aligment to the end of the session
    df['a_'+align] = df.fixed_times-df[align]

    # cluster_list = df_all.cluster_id.unique()
    df = df.sort_values('trial')

    y = df.groupby('trial')[decode].mean()
    
    # Filter for the spikes that occur in the interval we are analyzing
    df = df.loc[(df['a_'+align] > start) & (df['a_'+align] < stop)]

    df_final = pd.DataFrame()
    df_final = df.groupby(['trial', 'cluster_id']).count()
    df_final.reset_index(inplace=True)
    df_final = df_final.pivot_table(
        index=['trial'], columns='cluster_id', values='fixed_times', fill_value=0).rename_axis(None, axis=1)
    df_final = df_final.reindex(cluster_list, axis=1, fill_value=0)

    result = pd.merge(df_final, y, how="right", on=["trial"]).fillna(0)
    result = result.rename(columns={decode: "y"})
    result['y'] = np.where(result['y'] == 0, -1, result['y'])

    return result, result['y']

def train(df, decode='vector_answer', align='Delay_OFF', start=-0.5, stop=0, cluster_list = [], 
            ratio=0.65, test_index=[],  train_index=[], fakey=[], delay_only=False):

    df_final, y = interval_extraction(df,decode = decode, align = align, start = start, stop = stop, cluster_list = cluster_list, delay_only=delay_only)
    
    # This is mainly for the session shuffles
    if len(fakey) > 1:
        print('Using shuffled session')
        y = fakey[len(fakey)-len(y):]
        df_final['y'] = y   
        
    train_cols = df_final.columns
    
    #Train the model   
    df_final.reset_index(inplace=True)
    df_final = df_final.drop(columns ='trial')
        
    if len(test_index) >= 1:
        train = df_final.loc[train_index,:]
        test = df_final.loc[test_index,:]
        x_test = test.iloc[:, test.columns != 'y']
        y_test = test['y']
        x_train = train.iloc[:, train.columns != 'y']
        y_train = train['y']
        
    else:
        x_train = df_final.iloc[:, df_final.columns != 'y']
        y_train = df_final['y']
        x_test = x_train
        y_test = y_train

        
    #Normalize the X data
    # sc = RobustScaler()
    # sc_fit = sc.fit(x_train)
    # x_train = sc.fit_transform(x_train)
    # x_test = sc.fit_transform(x_test)
    
    model = LogisticRegression(solver='liblinear', penalty = 'l2', C=0.99, class_weight='balanced').fit(x_train, y_train)
    # model = LogisticRegression(solver='liblinear', penalty = 'l1', C=0.9).fit(x_train, y_train)

    train_cols = df_final.columns
    
    p_pred = model.predict_proba(x_test)    
    y_pred = model.predict(x_test)    
    f1score= f1_score(y_test, y_pred, average='weighted')

    y_test = np.where(y_test == -1, 0, y_test) 
    y_new = y_test.reshape(len(y_test), 1).astype(int)
    # y_new = y_test.values.reshape(len(y_test), 1).astype(int)
    score_ =  np.take_along_axis(p_pred,y_new,axis=1)   

    print('score:', np.mean(score_), 'f1_score ', f1score)
    
    return model, train_cols, np.mean(score_)

def test(df,model, epoch='Stimulus_ON',initrange=-0.4,endrange=1.5,r=0.2, train_cols=None, variable='ra_accuracy',
                      hit=1, nsurrogates = 100, decode='vector_answer', ratio=0, cluster_list = [], test_index=[], fakey=[], 
                        delay_only=False, score_options = 'standard'):
    '''
    Function that tests a previously trained function (func. train_decoder) on population activity of specific segments
    
    Attributes
        - df: DataFrame. it contains a whole ephys session without curation. 
        - WM and RL are the variables to consider a trial in the RL or in the WM-module. Both need to be floats. 
        - epoch: str. Moment at which the data will be aligned to. 
        - initrange: float. 
        - endrange: float.
        - r: float 
        - model. function. 
        - train_cols
        - name. String
        - variables. List. 
        - hits. List. 
        - colors. List
        - nsurrogates. Int. 
        - indexes. List 
        - decode. String
    
    Return
        - df_real
        - df_iter
        It will also make a plot. 
    '''
    
    df_real = pd.DataFrame()
    df_iter = pd.DataFrame(columns = ['iteration','score', 'times','epoch' ,'variable'])
    

    times = [] # Timestamps
    real_score = [] # real scoring of the decoded
    index_iter = 0
    print_value = True

    for start, stop in zip(np.arange(initrange,endrange-r,r),np.arange(initrange+r,endrange,r)):
        times.append((start+stop)/2)
        df_final, y = interval_extraction(df,decode = decode, align = epoch, start = start, stop = stop, cluster_list=cluster_list, delay_only=delay_only)
        
        # Sometimes the testing and the trainind dataset have different neurons since they are looking at different trials and perhaps there were no spikes
        # coming from all neurons. We compare which columns are missing and add them containing 0 for the model to work. 
        test_cols = df_final.columns
        
        common_cols = train_cols.intersection(test_cols)
        train_not_test = train_cols.difference(test_cols)
        for col in train_not_test:
            df_final[col] = 0

        #The other way round. When training in segmented data, sometimes the training set is smaller than the testing (for instance, when training in Hb trials and testing in WM)
        test_not_train = test_cols.difference(train_cols)
        for col in test_not_train:
            df_final.drop(columns=[col],inplace=True)
        
        # Reorder so we can use the fit from the trianing of the Robustscaler
        df_final = df_final.reindex(columns=train_cols)
        
        # This is for the session shuffles
        if len(fakey) > 1:
            print('Using shuffled session')
            y = fakey[len(fakey)-len(y):]
            df_final['y'] = y   
            
        #Train the model"
        if len(test_index) >= 1:
            # Split data in training and testing
            # x_train, x_test, y_train, y_test =\
            #     train_test_split(df_final, y, test_size=test_sample,random_state=random_state)
            
            df_final.reset_index(inplace=True)
            df_final = df_final.drop(columns ='trial')            
            test = df_final.loc[test_index,:]
            # print('Fold',str(fold_no),'Class Ratio:',sum(test['y'])/len(test['y']))
            x_test = test.iloc[:, test.columns != 'y']
            y_test = test['y']             

        else:
            x_train = df_final.iloc[:, df_final.columns != 'y']
            y_train = df_final['y']
            x_test = x_train
            y_test = y_train

        #Normalize the X data
        # sc = RobustScaler()
        # x_test = sc.fit_transform(x_test)
        # x_test = sc_fit.transform(x_test)

        p_pred = model.predict_proba(x_test)
        y_pred = model.predict(x_test)

        if score_options == 'standard':
            score_ = model.score(x_test, y_test)

        else:
            y_test = np.where(y_test == -1, 0, y_test)
            y_new = y_test.reshape(len(y_test), 1).astype(int)
            # y_new = y_test.values.reshape(len(y_test), 1).astype(int)
            score_ =  np.take_along_axis(p_pred,y_new,axis=1)

        real_score.append(score_)

        # f1score= np.mean(f1_score(y_test, y_pred, average='macro'))
        # real_score.append(f1score)

        # precision = np.mean(precision_score(y_test, y_pred))
        # recall = np.mean(recall_score(y_test, y_pred))
        if print_value == True:
            print(y_test.value_counts())
            print_value = False
        for i in np.arange(nsurrogates):
            y_perr = shuffle(y_test)

            if score_options == 'standard':
                score_ = model.score(x_test, y_perr)   
                score_  = np.mean(score_)
            else:
                # y_new = y_perr.values.reshape(len(y_perr), 1).astype(int)
                y_new = y_perr.reshape(len(y_perr), 1).astype(int)
                result =  np.take_along_axis(p_pred,y_new,axis=1)
                score_  = np.mean(result)
                        
            new_row = {'iteration': i, 'score': score_, 'times': (start+stop)/2, 'epoch' : epoch, 'variable' : str(variable)+'_'+str(hit)}
            
            # Use the `loc` indexer to insert the row
            df_iter.loc[index_iter] = new_row
            index_iter +=1 
            
    times.append('trial_type')
    real_score.append(variable+'_'+str(hit))
    a_series = pd.DataFrame([real_score], columns = times)
    df_real = pd.concat([df_real,a_series], ignore_index=True)
    
    return df_real, df_iter

def synch_trial(df, T, upper_plot, lower_plot, trial=0, start=-2, stop=0):
    """
    Synchronize and plot trial data for electrophysiological analysis.
    Parameters:
    df : pandas.DataFrame
        DataFrame containing the trial data.
    T : int
        Trial number to synchronize.
    upper_plot : matplotlib.axes.Axes
        Axes object for the upper plot.
    lower_plot : matplotlib.axes.Axes
        Axes object for the lower plot.
    trial : int, optional
        Trial index for the plot title (default is 0).
    start : int, optional
        Start time for the analysis window in seconds (default is -2).
    stop : int, optional
        Stop time for the analysis window in seconds (default is 0).
        
    Returns:
    None
    """
    dft = df.loc[df.trial ==T]
    align='Stimulus_ON'
    # stop = dft.delay.unique()[0]+5 # end of analyzed window
    delay = dft.delay.unique()[0]
    #Filter for the last 2 seconds of the ITI
    dft = dft.loc[(dft['a_'+align]>start)&(dft['a_'+align]<stop)]
    
    # Recover amount of neurons that were being registered at that trial interval
    n_neurons = len(df.cluster_id.unique())
    times_spikes = dft['a_'+align].values
    times_spikes = times_spikes*1000*ms #transform to ms
    
    ############################################################ Set the strat and end time of the train
    stop_time =  stop*1000*ms ## End of the trial in ms
    start_time = start*1000*ms ## Start of the trial in ms     
    
    ############################################################ Spiketrain
    spiketrain = SpikeTrain(times_spikes, units=ms, t_stop=stop_time, t_start=start_time) 
    
    ############################################################ 
    histogram_rate = time_histogram([spiketrain], 20*ms, output='rate')
    times_ = histogram_rate.times.rescale(s)
    firing_real = histogram_rate.rescale(histogram_rate.dimensionality).magnitude.flatten()
    
    real_std=np.std(firing_real) # Store the real std value
    
    #         t1_start = process_time() 
    list_std = []
    for i in range(surrogates):
        # Create a random shuffle for the same amount of spikes in that interval
        random_float_list = np.random.uniform(start, stop, len(times_spikes))
        surrogate_spikes = np.array(random_float_list)*1000*ms #transform to ms
        spiketrain = SpikeTrain(surrogate_spikes, units=ms, t_stop=stop_time, t_start=start_time) 
    
        histogram_rate = time_histogram([spiketrain], 20*ms, output='rate')
        times_ = histogram_rate.times.rescale(s)
        firing = histogram_rate.rescale(histogram_rate.dimensionality).magnitude.flatten()
    
        list_std.append(np.std(firing))
    
    # Organized by cluster_id and corrected for FR ____________________________
    cluster_id=[]
    FR_mean=[]
    
    for N in df.cluster_id.unique():
        spikes = dft.loc[dft.cluster_id==N]['a_'+align].values
        FR_mean.append(len(spikes)/abs(stop-start))
        cluster_id.append(N)
    
    df_spikes = pd.DataFrame(list(zip(cluster_id,FR_mean)), columns =['cluster_id','FR'])
    df_spikes = df_spikes.sort_values('FR')
    df_spikes['new_order'] = np.arange(len(df_spikes))
    
    dft = pd.merge(df_spikes, dft, on=['cluster_id'])
    
    print('Synch:', real_std/np.mean(list_std), '; WM:', str(dft.WM_roll.unique()[0]))
    
    panel = upper_plot
    panel.set_title(trial)
    j=0
    for N in dft.new_order.unique():
        spikes = dft.loc[dft.new_order==N]['a_'+align].values
        j+=1
        panel.plot(spikes,np.repeat(j, len(spikes)), '|', markersize=1, color='black', zorder=1)
    
    panel = lower_plot
    panel.plot(times_,firing_real/n_neurons*1000)
    # y = np.arange(0,j+1,0.1)
    # panel.fill_betweenx(y, cue_on,cue_off, color='grey', alpha=.4)
    # panel.fill_betweenx(y, cue_off+delay,cue_off+delay+.2, color='beige', alpha=.8)
    panel.set_ylim(0,15)
    panel.set_ylabel('Firing rate (spks/s)')