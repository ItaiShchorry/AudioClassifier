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
# ## Generic Exploration Data Analysis

# + [markdown] Collapsed="false"
# This main goal of this section is to **convince ourselves that the data we have is sufficient for our task**.   
#
# We'll do this by covering standard practices that might yield insights into our problem, but there's a lot of room for creativity & domain-related drill downs in this section.  
# An important rule  - **during this stage we should not process or change our data at all**.   
#
# We're simply trying to understand our data & record some observations, automatically & actively - changing it during that process might distort our understanding.  
# We'll have a section where we centralize the actions we're performing on the input data.

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
from functools import partial

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy
from scipy.cluster import hierarchy as hc
from pandas.api.types import is_integer_dtype
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GroupShuffleSplit
# import qgrid
# qgrid.set_grid_option('maxVisibleRows', 10)

#jupyter_code
# from IPython.display import display,HTML
# def disp(df,max_rows=30, max_cols=1000):
#     with pd.option_context("display.max_rows", max_rows):
#         with pd.option_context("display.max_columns", max_cols):
#             display(df)


# + [markdown] Collapsed="false"
# We'll start by setting some of the configurations relevant for our use case.  
#

# + Collapsed="false"
#export


# should probably be in a utils.py
def create_if_does_not_exist(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        


# + Collapsed="false"
#export

# method in DataCentral
def read_df_from_path(file_path):
    """
    Loading df into memory
    """
    file_path = Path(file_path)
    pos = file_path.as_posix()
    if pos.endswith(".pkl") or pos.endswith(".p"):
        with open(file_path, "rb" ) as f:
            df = pickle.load(f)

    elif pos.endswith(".csv"):
        df = pd.read_csv(file_path)
    
    else:
        print("not a valid format.")
        
    return df



# + [markdown] Collapsed="false"
# ### Create Summary View & Profiler
# We'll summarize some meta-level information + aggregative measures for each column

# + Collapsed="false"
#export

# class SummaryTable. Everything here will be methods/classes inside it, the create will be the init
def encapsulate_dtypes_to_one_column(summary_table):
    """
    Possible values:
    numeric
    cat
    time
    """
    summary_table['suggested_type_class'] = None
    summary_table.loc[summary_table['is_cat'], 'suggested_type_class'] = 'cat'
    summary_table.loc[summary_table['is_time_based'], 'suggested_type_class'] = 'time'
    summary_table['suggested_type_class'] = summary_table['suggested_type_class'].fillna('numeric')

    # grab original place of is_cat and drop redundancies
    summary_table['is_cat'] = summary_table['suggested_type_class'].copy()
    summary_table.drop(['suggested_type_class', 'is_time_based'], axis=1, inplace=True)
    summary_table.rename({'is_cat': 'suggested_type'},axis=1,inplace = True)
    return summary_table



def add_missing_data_columns(summary_table, df):
    summary_table['missing_total'] = df.isnull().sum().reset_index(name='col_name')['col_name']
    summary_table['missing_percent'] = (df.isnull().sum()/df.isnull().count()).reset_index(name='col_name')['col_name'] * 100
    return summary_table

def init_summary_numeric_cols(summary_table):
    summary_table['max'] = None
    summary_table['min'] = None
    summary_table['mean'] = None
    summary_table['median'] = None
    summary_table['std'] = None
    
    return summary_table

def calculate_summary_numeric_cols(summary_table, df):
    is_numeric = summary_table['suggested_type'] == 'numeric'
    
    # handle case where we update statistics - only update existing columns
    if 'should_drop' in summary_table.columns:
        is_numeric = is_numeric & ~summary_table['should_drop']
    
    summary_table.loc[is_numeric,'mean'] = summary_table[is_numeric]['col_name'].apply(lambda x: (df[x].mean()))
    summary_table.loc[is_numeric,'median'] = summary_table[is_numeric]['col_name'].apply(lambda x: (df[x].median()))
    summary_table.loc[is_numeric,'std'] = summary_table[is_numeric]['col_name'].apply(lambda x: (df[x].std()))
    summary_table.loc[is_numeric,'min'] = summary_table[is_numeric]['col_name'].apply(lambda x: (df[x].min()))
    summary_table.loc[is_numeric,'max'] = summary_table[is_numeric]['col_name'].apply(lambda x: (df[x].max()))

    return summary_table

def add_frequency_based_statistics(summary_table, df):
    is_cat = summary_table['suggested_type'] == 'cat'
    
    # handle case where we update statistics - only update existing columns
    if 'should_drop' in summary_table.columns:
        is_cat = is_cat & ~summary_table['should_drop']
    
    cat_cols = summary_table.loc[is_cat, 'col_name'].tolist()
    for c in cat_cols:
        val_cnts = df[c].value_counts()
        max_str_format = str(val_cnts.index[0]) + ' ({0:.2f}%)'.format(100*val_cnts.iloc[0]/df.shape[0])
        min_str_format = str(val_cnts.index[-1]) + ' ({0:.2f}%)'.format(100*val_cnts.iloc[-1]/df.shape[0])

        summary_table.loc[summary_table['col_name'] == c, 'max'] = max_str_format
        summary_table.loc[summary_table['col_name'] == c, 'min'] = min_str_format

    return summary_table


def create_col_level_summary_table(df):
    """
    Summary statistics per column.
    This summary can already give us some insights on the data and actions that we'd like to take, e.g. drill-down a high-cardinality column or drop one with many missing values.   
    
    TODO itai -
    - add some relationships with the target variable (frequency, correlation)
    """
    # Create summary level data
    summary_table = df.T.reset_index().rename(columns={'index': 'col_name'})
    summary_table = summary_table.drop((set(summary_table.columns) - {'col_name'}), axis=1)
    
    # add current dtype
    summary_table['current_dtype'] = summary_table['col_name'].apply(lambda x: str(df[x].dtype))
    
    #### start building suggested_dtype ###
    # divide between categorical and numerical. Very simplistic, validate result
    summary_table['is_cat'] = summary_table['col_name'].apply(lambda x: str(df[x].dtype) in ('object','category')) 
    
    # defaults
    summary_table['is_time_based'] = summary_table['col_name'].str.lower().str.contains('|'.join(['time','year','month','week','day','hour','timestamp','date'])) 
    summary_table['is_time_based'] = summary_table['is_time_based'] & summary_table['col_name'].apply(lambda x: str(df[x].dtype) in ('object')) # TODO Doesn't function as expected
    summary_table['is_time_based'] = summary_table['is_time_based'] | summary_table['col_name'].apply(lambda x: str(df[x].dtype) in ('datetime64[ns]','datetime32[ns]'))
    summary_table['module'] = None # Essentially the suite name for each test. TODO itai automate according to column names formatting
    summary_table['is_knob'] = False # go over main features with domain expert
    
    # num of unique values
    summary_table['num_unique_values'] = summary_table['col_name'].apply(lambda x: len(df[x].unique()))
    
    # encapsulate types
    summary_table = encapsulate_dtypes_to_one_column(summary_table)
    
    # Numeric statistics + categorical frequency info
    summary_table = init_summary_numeric_cols(summary_table)
    summary_table = calculate_summary_numeric_cols(summary_table, df)
    summary_table = add_frequency_based_statistics(summary_table, df)
    
    # Calculate nulls statistics
    summary_table = add_missing_data_columns(summary_table, df)

    summary_table = initialize_dropped_columns_tracking(summary_table)
    return summary_table


    
def update_summary_table_stats(df, summary_table):
    """
    Use current state of df to recalculate statistics of summary table.
    Will only update columns that aren't going to be dropped.
    """
    summary_table = add_frequency_based_statistics(summary_table, df)
    summary_table = calculate_summary_numeric_cols(summary_table, df)
    
    return summary_table


def initialize_dropped_columns_tracking(summary_table):
    """
    We'll hold a separate dataFrame for dropping columns & maintaining an understanding on why we dropped them
    """
    summary_table['should_drop'] = False
    summary_table['reason_to_drop'] = ""
    return summary_table

def get_dtype_categories_from_summary(summary_table):
    # We only care about columns that we didn't decide to remove
    existing_cols = summary_table.loc[~summary_table['should_drop']]
    
    cat_cols = existing_cols.loc[existing_cols['suggested_type'] == 'cat', 'col_name'].tolist()
    time_cols = existing_cols.loc[existing_cols['suggested_type'] == 'time', 'col_name'].tolist()
    numeric_cols = existing_cols.loc[existing_cols['suggested_type'] == 'numeric', 'col_name'].tolist()
    return {'cat': cat_cols, 'time': time_cols, 'numeric': numeric_cols}



# ### Exploratory Visualizations
def save_sns_viz(viz_name, viz_plot, viz_path):
    """
    saves a visualization under the appropriate folder
    """
    save_path = viz_path/viz_name
    try:
        viz_plot.savefig(save_path)
    
    except Exception: # sometimes we need to grab the figure
        try:
            fig = viz_plot.get_figure() 
            fig.savefig(save_path)
        
        except Exception:
            print("failed to save visualization for object type " + str(type(viz_plot)) + ". Try manually saving it")


#export
# methods in class EDAer
def pearson(x,y):
    corr, p_val = scipy.stats.pearsonr(x,y)
    return corr, p_val

def spearman(x,y):
    corr, p_val = scipy.stats.spearmanr(x,y)
    return corr, p_val

def kendall(x,y):
    corr, p_val = scipy.stats.kendalltau(x,y)
    return corr, p_val

def heatmap(df):
    return sns.heatmap(df.corr(method='spearman',min_periods=int(df.shape[0]*0.5)))


# + Collapsed="false"
#export

# method in class DataCentral
def change_dtype(df, summary_table, col, to_type):
    """
    Changes datatype for column in a traceable manner
    """
    # change type in dataframe
    if to_type == "category":
        df[col] = pd.Categorical(df[col])
    else:
        df[col] = df[col].astype(to_type)
    
    # update summary for future usage
    if (to_type != 'category') and np.issubdtype(to_type, np.number):
        summary_table.loc[summary_table['col_name'] == col, 'suggested_type'] = 'numeric'
        summary_table.loc[summary_table['col_name'] == col, 'current_dtype'] = str(to_type)
        
    else: 
        summary_table.loc[summary_table['col_name'] == col, 'suggested_type'] = 'cat'
        summary_table.loc[summary_table['col_name'] == col, 'current_dtype'] = str(to_type)
        
        if to_type in ('datetime64[ns]','datetime32[ns]'):
            summary_table.loc[summary_table['col_name'] == col, 'suggested_type'] = 'time'
            
    return df, summary_table


def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed) # multi-GPU


def drop_columns(proc_df, summary_table, cols_to_drop, reason):
    if cols_to_drop:
        summary_table.loc[summary_table['col_name'].isin(cols_to_drop), 'should_drop'] = True
        summary_table.loc[summary_table['col_name'].isin(cols_to_drop), 'reason_to_drop'] = reason

        proc_df = proc_df.drop(cols_to_drop,axis=1,errors='ignore')

    return proc_df, summary_table




# + [markdown] Collapsed="false"
# ### Outlier Detection
# def drop_outliers(proc_df, summary_table, remove_outliers=False, target=None, outlier_sig_cutoff=None, config=None):
#     """
#     Automatically drops outliers according to standardization and sigma-distance for each column
#     """
#     
#     if config is not None:
#         remove_outliers= config['PREPROCESSING_CONFIG']['REMOVE_OUTLIERS']
#         target= config['DATA_CONFIG']['TARGET']
#         summary_table= config['DATA_CONFIG']
#         outlier_sig_cutoff= config['PREPROCESSING_CONFIG']['OUTLIER_SIGMA_CUTOFF']
#     
#     if remove_outliers:
#         outlier_df = proc_df.copy().drop(target,axis=1)
#
#         for c in proc_df.drop(target,axis=1).columns:
#             mean = summary_table.loc[summary_table['col_name']== c,'mean'].values[0]
#             std = summary_table.loc[summary_table['col_name']== c,'std'].values[0]
#
#             cut_off = std * outlier_sig_cutoff
#             lower, upper = mean - cut_off, mean + cut_off
#             outlier_df[c] = ~((lower <= outlier_df[c]) & (outlier_df[c] <= upper ))
#
#
#         outliers = outlier_df[outlier_df.any(axis=1)].index.to_list()
#         proc_df = proc_df.loc[~proc_df.index.isin(outliers)]
#         print(f"dropped {len(outliers)} outliers")
#     
#     return proc_df

# + [markdown] Collapsed="false"
# ### Data Imputation

# + [markdown] Collapsed="false"
# Methods to fill missing values

# + Collapsed="false"

# Future subtle point to consider - moving the normalization before the imputation logic (missing values will amplify certain values incorrectly)
# Alternatively, remember uuids for missing values and avoid them in calculations
def robust_sigma(x, axis=None, rng=(25, 75), scale='normal', nan_policy='omit', interpolation='nearest', keepdims=False):
    import scipy.stats
    rs = scipy.stats.iqr(x, axis=axis, rng=rng, scale=scale, nan_policy=nan_policy, interpolation=interpolation, keepdims=keepdims)
    return (rs)

def z_score_calc(x, config, override=True, robust=True):
    """
    Calculates the z score of the passed `pd.Series`
    by default the function calculates the robust metric with median and robust sigma
    """
    # produce relevant stats
    if not override:
        x_median = config['NORMALIZATION_STATS'][x.name]['X_MEDIAN']
        
        if robust:
            x_rs = config['NORMALIZATION_STATS'][x.name]['X_ROB_SIG']
        else:
            x_mean = config['NORMALIZATION_STATS'][x.name]['X_MEAN']
            x_std = config['NORMALIZATION_STATS'][x.name]['X_STD']
    
    else:
        if robust:
            x_rs = robust_sigma(x)
        else:
            x_mean = x.mean()
            x_std = x.std()
            
    
    ### calculate operation
    if robust:
        if x_rs == 0:
                return x + 0.5
        z_scores = (x - x_median)/x_rs
    else:
        if x_std == 0:
                return x + 0.5
        z_scores = (x - x_mean)/x_std
    
    # log stats for future inferencing
    if override:
        config['NORMALIZATION_STATS'][x.name] = dict()
        config['NORMALIZATION_STATS'][x.name]['X_MEDIAN'] = x_min

        if robust:
            x_rs = config['NORMALIZATION_STATS'][x.name]['X_ROB_SIG']
        else:
            config['NORMALIZATION_STATS'][x.name]['X_MEAN'] = x_mean
            config['NORMALIZATION_STATS'][x.name]['X_STD'] = x_std
            
    return z_scores

def minmax_scale(x, config, override=True):
    
    # produce relevant stats
    if not override:
        x_min = config['NORMALIZATION_STATS'][x.name]['X_MIN']
        x_max = config['NORMALIZATION_STATS'][x.name]['X_MAX']
    
    else:
        x_max = x.max()
        x_min = x.min()
    
    ### calculate operation
    
    # if only one value in series return constant 0.5
    if x_max == x_min:
        if x_max > 0 or x_max < 0: 
            return x/x * 0.5 
        else:
            return x + 0.5

    x -= x_min
    x /= (x_max - x_min)
    
    if override:
        # log stats for future inferencing
        config['NORMALIZATION_STATS'][x.name] = dict()
        config['NORMALIZATION_STATS'][x.name]['X_MIN'] = x_min
        config['NORMALIZATION_STATS'][x.name]['X_MAX'] = x_max
    
    return x

def normalize(df, config, summary_table, override=True):
    
    mapping = config['NORMALIZATION_MAPPING']
    norm_method = config['NORMALIZATION_METHOD']
    perf_norm = config['PERFORM_NORMALIZATION']
    
    if override:
        config['NORMALIZATION_STATS'] = dict()  
    
    if not perf_norm:
        return df
        
    type2cols_dict = get_dtype_categories_from_summary(summary_table) # Temporary solution, this probably shouldn't be calculated everytime we care about the dtypes
    for key, value in mapping.items():
        groups = value['groups']
        columns = value['columns']
        norm_columns =  list(set(type2cols_dict['numeric']) & set(columns)) if columns else type2cols_dict['numeric']

        if norm_method == 'minmax':
            partial_minmax_scale = partial(minmax_scale, config=config, override=override)
            df[norm_columns] = df.groupby(groups)[norm_columns].transform(partial_minmax_scale) if groups else df[norm_columns].transform(partial_minmax_scale)

        elif norm_method == 'zscore':
            partial_z_score_calc = partial(z_score_calc, config=config, override=override)
            df[norm_columns] = df.groupby(groups)[norm_columns].transform(partial_z_score_calc) if groups else df[norm_columns].transform(partial_z_score_calc)

    if df.isnull().sum().sum() > 0 :
        print("You got null values, check cardinality in your groups")
    
    return df, config



# + Collapsed="false"
#export
def split_data(proc_df, config):
    
    X_train, X_, y_train, y_ = train_test_split(proc_df.drop(config['TARGET'], axis=1),  proc_df[config['TARGET']], train_size=config['TRAIN_SIZE'], random_state=config['RANDOM_STATE'], stratify=proc_df[config['TARGET']])
    print (f"train: {len(X_train)} ({(len(X_train) / len(proc_df)):.2f})\n"
           f"remaining: {len(X_)} ({(len(X_) / len(proc_df)):.2f})")

    # Split to test
    X_valid, X_test, y_valid, y_test = train_test_split(X_, y_, train_size=round(config['VALID_SIZE']/(config['VALID_SIZE']+config['TEST_SIZE']), 2), stratify=y_, random_state=config['RANDOM_STATE'])
    
    return X_train, X_valid, X_test,y_train, y_valid, y_test


# + Collapsed="false"
#export
# methods under DataCentral. Create a flag of done_preprocessing to avoid bad usage of these 2 methods
def get_dtype_categories_from_df(proc_df):
    """
    Notice we should only use this method on the fully-preprocessed dataframe,
    i.e. one that completely handled all the data types transformations
    """
    dt_df = pd.DataFrame(proc_df.dtypes, columns=['dtype'])
    
    # fill relevant values
    dt_df['type_category'] = 'numeric'
    dt_df.loc[dt_df['dtype'] == 'category', 'type_category'] = 'cat'
    dt_df.loc[dt_df['dtype'].isin(['datetime64[ns]','datetime32[ns]']), 'type_category'] = 'time'
    dt_df = dt_df.drop('dtype', axis=1)
    
    # transform to dtypes dict
    col2type_dict = dt_df.to_dict()['type_category']
    type2col_dict = {'cat': [], 'numeric': [], 'time': []}
    for k,v in col2type_dict.items():
        type2col_dict[v].append(k)
    
    return type2col_dict

def one_hot_encode_dataset(X_train, X_test=None):
        # TODO !!!!!THIS IS A TEMPORARY SOLUTION!!!!!
        # the categorical possibilities should be learned from the train data only
        # and the test data should be transformed according to the train-data transformations
        # meaning if there's a value that exists only in the test data, it should go into an UNK value.
        # Another thing is to notice we've already changed the feature names in this point
        
        X_train_final = X_train.copy()
        
        if X_test is not None:
            X_test_final = X_test.copy()
        
        # concat dfs
        X_train_final['origin'] = 'train'
        
        if X_test is not None:
            X_test_final['origin'] = 'test'
            X = pd.concat([X_train_final, X_test_final])
            
        else:
            X = X_train_final
        
        # infer type groups
        type2cols_dict = get_dtype_categories_from_df(X)
        cat_cols = type2cols_dict['cat']
        
        if cat_cols:
            X = pd.get_dummies(X,columns=cat_cols)

        # split again to train & test
        X_train_final = X.loc[X['origin'] == 'train']
        X_train_final = X_train_final.drop('origin', axis=1)
        
        if X_test is not None:
            X_test_final = X.loc[X['origin'] == 'test']
            X_test_final = X_test_final.drop('origin', axis=1)
        
        else:
            X_test_final = None
            
        return X_train_final, X_test_final



# + Collapsed="false"
#export
# method under DataCentral
def run_data_module(config, df=None, prod=False):
    """
    A single method that receives the input dataframe and outputs ready inputs for the model, according to the configuration file and the summary table
    prod=True means it is running in production, i.e. we don't have a y value
    """
    
    create_if_does_not_exist(config['ANALYSIS_RESULT_FOLDER'])
    
    if df is None:
        df = read_df_from_path(config['FILE_PATH'])
    
    if prod:
        summary_table = read_df_from_path(config['SUMMARY_TABLE_PATH'])
        proc_df = df.copy()
        
    else:
        summary_table = create_col_level_summary_table(df)
        summary_table = initialize_dropped_columns_tracking(summary_table)
    
    proc_df,config = normalize(proc_df, config=config, summary_table=summary_table, override=not prod)
    
    return proc_df




# + [markdown] Collapsed="false"
# ### Hierarchical Clustering

# + Collapsed="false"
#export
# wherever we put the clustering methods from the EDA, this should go to the same place

import scipy.stats
import pandas as pd
from scipy.cluster import hierarchy as hc
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering


def hierarchical_clustering(dist_matrix, linkage_method='ward', fcluster_th = 0.7, fcluster_criterion = 'distance', print_dendrogram = False):
    
    dist_matrix_condensed = scipy.cluster.hierarchy.distance.squareform((1-dist_matrix))
    z                     = scipy.cluster.hierarchy.linkage(dist_matrix_condensed, method=linkage_method)
    n_clusters            = len(np.unique(scipy.cluster.hierarchy.fcluster(z, t=fcluster_th, criterion=fcluster_criterion)))
    
    if print_dendrogram:
        fig = plt.figure(figsize=(10,6))
        scipy.cluster.hierarchy.dendrogram(z, labels = dist_matrix.columns)
        plt.show()
        
    return n_clusters



def picklel(p):
    with open(p,'rb') as f:
        x = pickle.load(f)
    
    return x

def pickles(p,x):
    with open(p,'wb') as f:
        pickle.dump(x, f)
        
def get_subdirectories(main_dir):
    return [name for name in os.listdir(main_dir)
            if os.path.isdir(os.path.join(main_dir, name))]