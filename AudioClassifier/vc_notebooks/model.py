# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: fastai2
#     language: python
#     name: fastai2
# ---

# + [markdown] Collapsed="false"
# # Model
# The code here will be mainly under MLResearcher, a class that will mainly contain:
# 1. DataValidator - the only example I currently have is the data distribution comparisons but there might be more meat in here.
# 2. ExperimentManager that executes training & tracks performance.
#     - ExpMang will have a ModelOptimizer class inside that contains all different operations we can perform to try & improve results
#       Those will include both model hyperparameters but also different data operations, such as data balancing, normalizations etc.
#       In order to do that, ExpMang will also have a DataCentral object inside it
#
# 3. Analyzer - results analyzer

# + Collapsed="false"
#export
from pathlib import Path
import pickle
import copy
import os
import json
import logging
from datetime import datetime
from enum import Enum
import random
import re
import warnings

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy
from scipy.cluster import hierarchy as hc
from pandas.api.types import is_integer_dtype

import sklearn
from sklearn import tree
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from dtreeviz.trees import dtreeviz
warnings.filterwarnings("ignore", category=FutureWarning)

# won't work, need to handle
from data import *


# + [markdown] Collapsed="false"
# ## Compare Data Distributions

# + [markdown] Collapsed="false"
# From a training perspective, this suite can be used to compare train & validation distributions to make sure we do not have data leakage.  
# Practically, we'll use it after we've split the data, assuming that the train & validation data represent the same distribution, i.e. "the real distribution" of our data.   
# To answer that question, we'll train a model **to predict whether a unit belongs to the training data or the validation data**.  
#
# For features that were identified as 'leaky' by our model, we'll run a [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test), a statistical operation to compare distributions.

# + Collapsed="false"
#export

# all methods here should be under class DistributionComparer (or something more catchy).
# I think this is worth to have his own class, because there's room to evolve it + the code here isn't small
# notice that we shouldn't call this Monitor or something similar, because that's 1 use of these methods, but there are many (OOD, compare train-valid etc.)


# Methods for distribution shifts tests
from scipy.stats import ks_2samp
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
# from xgboost import XGBClassifier
import pickle
# from downsemble.classification import DSClassifier
import shap
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from termcolor import colored

def kolmogorov_smirnov(x, y, impacting_cols = None, run_all_columns = False):
    
    df_ks = pd.DataFrame()
    if run_all_columns: 
        statistic, pvalue = zip(*[ks_2samp(x[col].to_numpy(), y[col].to_numpy()) for col in x.columns])
        df_ks = pd.DataFrame({'statistic': tuple([round(x,4) for x in statistic]), 'pvalue': tuple([round(x,4) for x in pvalue])}, index =  impacting_cols).sort_values('pvalue')
 
    if impacting_cols:
        statistic, pvalue = zip(*[ks_2samp(x[col].to_numpy(), y[col].to_numpy()) for col in impacting_cols])
        df_ks = pd.DataFrame({'statistic': tuple([round(x,4) for x in statistic]), 'pvalue': tuple([round(x,4) for x in pvalue])}, index =  impacting_cols).sort_values('pvalue')
    
    return df_ks

def out_of_domain_rf(x_train, x_valid , threshold, rand_state):
    def rf_feat_importance(m, df):
        return pd.DataFrame({'col':df.columns, 'feature_importances_rf':m.feature_importances_}
                       ).sort_values('feature_importances_rf', ascending=False).reset_index(drop=True)

    def rf(xs, y, n_estimators=40, max_features=0.8, min_samples_leaf=5, **kwargs):
        return RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,
            max_features=max_features, min_samples_leaf=min_samples_leaf).fit(xs, y)
    
    # Create is_validation dataset
    df_dom = pd.concat([x_train, x_valid])
    is_valid = np.array([0]*len(x_train) + [1]*len(x_valid))
    is_valid_data = pd.concat([df_dom, pd.DataFrame(is_valid, index=df_dom.index, columns=['is_valid'])], axis = 1)
    
    # run the model
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(is_valid_data.drop('is_valid', axis=1),  is_valid_data['is_valid'], test_size=0.33, random_state=rand_state)
    model = rf(X_train_rf, y_train_rf)
    
    # evaluations
    accuracy = metrics.accuracy_score(y_test_rf,  model.predict(X_test_rf).round())
    fi =  rf_feat_importance(model, is_valid_data.drop('is_valid', axis=1))
    
    return accuracy < threshold, accuracy, fi, X_test_rf[[fi.loc[0]['col']]], y_test_rf
    
def plot_hist(X_train, X_valid, different_columns_list):
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_train['data'] = 'train'
    X_valid['data'] = 'new'
    if len(different_columns_list) == 1:
        fig, ax = plt.subplots(ncols=2, figsize=(15,7))
        sns.histplot(pd.concat([X_train, X_valid]), x=different_columns_list[0], hue = 'data', ax=ax[0], bins = 30)
        sns.ecdfplot(pd.concat([X_train, X_valid]), x=different_columns_list[0], hue = 'data', ax=ax[1])
    else:
        fig, ax = plt.subplots(len(different_columns_list), 2, figsize=(16,5*len(different_columns_list)))
        for col, i in zip(different_columns_list, range(len(different_columns_list))):
            sns.histplot(pd.concat([X_train, X_valid]), x=col, hue = 'data', ax=ax[i, 0], bins = 30)
            sns.ecdfplot(pd.concat([X_train, X_valid]), x=col, hue = 'data', ax=ax[i, 1])
    return fig

def statistics_table(X_train, X_valid, different_columns_list):
    return pd.concat([X_train[different_columns_list].add_suffix('_train').describe(), X_valid[different_columns_list].add_suffix('_new').describe()], sort=False, axis=1).sort_index(axis=1)

def compare_train_valid_populations(X_train, X_valid, threshold_rf = 0.7, min_fi_for_running_ks=0.1, ks_required_pvalue = 0.05, threshold_dt = 0.5, rand_state=42):
    type2cols_dict = get_dtype_categories_from_df(X_train)
    cat_cols = type2cols_dict['cat']
    
    X_train, X_valid = one_hot_encode_dataset(X_train, X_valid)
    
    flag, accuracy, fi, X_test_rf, y_test_rf = out_of_domain_rf(X_train, X_valid, threshold_rf, rand_state)
    impacting_cols = fi.loc[fi['feature_importances_rf'] >= min_fi_for_running_ks, 'col'].tolist()
    
    print(f'Warning - Model was able to classify between train and validation with accuracy of {accuracy.round(2)}>{threshold_rf} (our threshold)')
    print(f'Significant impacting features were {impacting_cols}')
    print("Running Kolmogorov Smirnov on them features to compare train and validation numeric distributions")
    print("Note: in case the list is empty, it is likely that the contribution for the model is distributed between many features")


    # run kolmogorov-smirnov on impacting features
    df_ks = kolmogorov_smirnov(X_train, X_valid, impacting_cols)
    
    failed_ks_stats = []
    if not df_ks.empty:
        failed_ks_stats = df_ks.loc[df_ks['pvalue'] <= ks_required_pvalue]
        failed_ks_col_names = failed_ks_stats.index
    
    # initialize results df
    res_df = fi.set_index('col')[['feature_importances_rf']].copy()
    res_df['ks_statistic_res'] = None
    res_df['ks_pvalue'] = None
    
    if not df_ks.empty:
        # udpate df with ks failures
        res_df.loc[df_ks.index.tolist(), 'ks_statistic_res'] = df_ks['statistic']
        res_df.loc[df_ks.index.tolist(), 'ks_pvalue'] = df_ks['pvalue']
    
    if len(failed_ks_stats)>0:
        print(colored('KS test: Failed. The tests that failed are:', 'red'))
        print(colored(', '.join(failed_ks_col_names), 'red'))
        
        # visualize failures
        plot_hist(X_train, X_valid, failed_ks_col_names).show()
        display(statistics_table(X_train, X_valid, failed_ks_col_names))
    
    else:
        print(colored(f'KS test: All columns passed Successfully', 'green'))
    
    
    
    return res_df.sort_values('feature_importances_rf', ascending = False)



# + [markdown] Collapsed="false"
# **What can we do in case we're seeing a significant difference?**
# We'll start with error analysis - what features were indicative in separating the train & valid + possible drill-downs to why.  
# This will probably lead to one of the following options:
# - Re-evaluate our splitting method - Maybe our splitting method is not correct for this problem. e.g. we decided on random splitting, but we should split based on wafers
# - "Fix" difference - Perform additional preprocessing for the data to minimize the difference between train & valid distributions
# - Drop leaky features - the cleanest approach, as we nullify differences between train & valid
#     - If decided on the latter, remember to do this in a traceable manner using our **traceable_column_dropping method**

# + [markdown] Collapsed="false"
# ## Handle Imbalanced Data


# + [markdown] Collapsed="false"
# Let's try running a **partial dependence plot**.  
#
# This method utilizes our predictive model to gain the "pure" effect of a feature on the target variable, i.e. **approximate an f: feature -> target** that is neglecting the effects of other variables.  
# **The method**: We take  randomly chosen datapoints, and replace the plotted feature with iteratively-incremented values, **leaving all other features as they were**. We predict the target variable according to those, and produce f according to the aggregated results.
#

# + Collapsed="false"
#export
from sklearn.inspection import plot_partial_dependence

# should be under analyzer. Notice the analyzer will need access to MLResearcher.
def partial_dependence(exp_mang,features):
    if features:
        fig,ax = plt.subplots(figsize=(12, 4))
        plot_partial_dependence(exp_mang.best_model, exp_mang.best_model.X_train, KNOBS_LIST,
                                grid_resolution=20, ax=ax)
        



# + [markdown] Collapsed="false"
# **What can we do with this information?**  
# This question is analogous to what we can do with a predictive model:
# - Use the predictions directly
# - Help us change how we do business
#
#

def grab_model(mdl_name):
    """
    Given a model name from the W&B's UI, returns that model
    """
    
    _, permutations_dicts = generate_hp_search_params()
    
    perm_idx = int(mdl_name.split('_')[-1])
    permute_dict = permutations_dicts[perm_idx]

    learn = tabular_learner(dls, metrics=accuracy,cbs=[], model_dir=mdl_name, **permute_dict)
    learn.load(Path.cwd()/mdl_name/'model')
    
    return learn


import itertools
def generate_hp_search_params():
    """
    Currently hard-coded, should be modularized in next iterations
    """
    # create data hp combinations
    data_hparams = [
        {'REMOVE_OUTLIERS': True, 'OUTLIER_SIGMA_CUTOFF': 5},
        {'REMOVE_OUTLIERS': True, 'OUTLIER_SIGMA_CUTOFF': 8},
        {'REMOVE_OUTLIERS': True, 'OUTLIER_SIGMA_CUTOFF': 12},
        {'REMOVE_OUTLIERS': False}
         ]

    # create model hp combinations
    model_hparams = {
        'lr': [0.0001,0.005,0.001],
        'moms': [(0.95,0.85,0.95), (0.85,0.90,0.95), (0.95,0.90,0.85), (0.90,0.90,0.90)],
        'wd': [None, 0.1, 0.01]

    } 

    keys, values = zip(*model_hparams.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"created {len(permutations_dicts)} permutations")
    
    return data_hparams, permutations_dicts


from fastai.tabular.all import *
import wandb

def run_hp_search(global_config, df):
    data_hparams, permutations_dicts = generate_hp_search_params()
    
    for dh in data_hparams:
        run_config = global_config.copy()
        run_config.update(dh)

        # simple name to identify data operations
        run_name = dh['OUTLIER_SIGMA_CUTOFF'] if dh['REMOVE_OUTLIERS'] else 0
        run_name = f"cutoff_{run_name}"

        # preprocess data
        proc_df = run_data_module(run_config, df)
        X_train, X_valid, X_test,y_train, y_valid, y_test = split_data(proc_df, run_config)
        dls = create_dataloaders(X_train, X_valid,y_train, y_valid, run_config['TARGET'])

        # for every hp combination, run training run & log best model
        for i, hp in enumerate(permutations_dicts):
            run_training_run(dls, hp, f"tuning_{run_name}_{i}")

            
class PermutationImportance():
    """"
    Implementation by fast.ai superhero Zachary Mueller from his "walkwithfastai" project.
    Calculate and plot the permutation importance.
    """
    def __init__(self, learn:Learner, df=None, bs=None):
        "Initialize with a test dataframe, a learner, and a metric"
        self.learn = learn
        self.df = df
        bs = bs if bs is not None else learn.dls.bs
        if self.df is not None:
            self.dl = learn.dls.test_dl(self.df, bs=bs)
        else:
            self.dl = learn.dls[1]
        self.x_names = learn.dls.x_names.filter(lambda x: '_na' not in x)
        self.na = learn.dls.x_names.filter(lambda x: '_na' in x)
        self.y = dls.y_names
        self.results = self.calc_feat_importance()
        self.plot_importance(self.ord_dic_to_df(self.results))

    def measure_col(self, name:str):
        "Measures change after column shuffle"
        col = [name]
        if f'{name}_na' in self.na: col.append(name)
        orig = self.dl.items[col].values
        perm = np.random.permutation(len(orig))
        self.dl.items[col] = self.dl.items[col].values[perm]
        metric = learn.validate(dl=self.dl)[1]
        self.dl.items[col] = orig
        return metric

    def calc_feat_importance(self):
        "Calculates permutation importance by shuffling a column on a percentage scale"
        print('Getting base error')
        base_error = self.learn.validate(dl=self.dl)[1]
        self.importance = {}
        pbar = progress_bar(self.x_names)
        print('Calculating Permutation Importance')
        for col in pbar:
            self.importance[col] = self.measure_col(col)
        for key, value in self.importance.items():
            self.importance[key] = (base_error-value)/base_error #this can be adjusted
        return OrderedDict(sorted(self.importance.items(), key=lambda kv: kv[1], reverse=True))

    def ord_dic_to_df(self, dict:OrderedDict):
        return pd.DataFrame([[k, v] for k, v in dict.items()], columns=['feature', 'importance'])

    def plot_importance(self, df:pd.DataFrame, limit=20, asc=False, **kwargs):
        "Plot importance with an optional limit to how many variables shown"
        df_copy = df.copy()
        df_copy['feature'] = df_copy['feature'].str.slice(0,25)
        df_copy = df_copy.sort_values(by='importance', ascending=asc)[:limit].sort_values(by='importance', ascending=not(asc))
        ax = df_copy.plot.barh(x='feature', y='importance', sort_columns=True, **kwargs)
        for p in ax.patches:
            ax.annotate(f'{p.get_width():.4f}', ((p.get_width() * 1.005), p.get_y()  * 1.005))