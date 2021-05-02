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


class DTCategory(Enum):
    """
    Data type category. TODO replace all hard-coded usages with this enum
    """
    CAT = 'cat'
    NUMERIC = 'numeric' 
    TIME = 'time'

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



# + Collapsed="false"
# method in DataCentral
def run_pandas_profiling(df):
    """
    Create a profiling report based on pandas-profiling package.
    Main points to consider:
    - It might be infeasible & messy to run it on a very large dataset with thousands of tests. **Consider reducing to a few dozen columns**.
    - **Next steps** - It might be a good idea to explore the possibility of sending a config to the ProfileReport call to fit our demands. We'll basically want to compute things but avoid their visualization, which is the bottleneck
    - **Warnings section** - diverse set of warnings regarding abnormalities in our data. 
    - **Correlation section** - all kinds of correlations are comfortably calculated between all variables. See additional explanation on correlations in the next section
    - **Missing values section** - the interesting part here is viewing combination of missing values, i.e. feature_i and feature_j are frequently missed together. The dendrogram is a nice view for it
    """
    profile = ProfileReport(df, title='profiling report') 
    return profile


# + [markdown] Collapsed="false"
# ### Exploratory Visualizations

# + [markdown] Collapsed="false"
# Some relevant plots to stir your mind:
# - **Line plots**
# - **Scatter plots**
# - **Histogram** - binning of numerical values.
# - **Bar plot** - the height is the expectation, the line shows 95% confidence intervals for the mean
# - **Box plot** - useful for comparing categories + visualizing outliers
# - **Violin plot** - Same as boxplot, with the addition of seeing the distribution for Q1-Q3. Can draw 2 distributions from both sides of the plot
# - **Point plot** - Slimmer presentation of boxplots, making it easier for the eye to compare slopes
# - **Heatmap** - Frequently used for correlations. See example in the next section  
# - **Relative plots (relplot)** - Scatter & line sub-plots for each value in a categorical variable
# - **Categorical plots (catplot)** - same as replot, but for categoricals
#
# We've provided examples for these using [Seaborn](https://seaborn.pydata.org/). Notes:
# - Take a look at the Seaborn  documentation to utilize them to their fullest. Some recommendations:
#     - general - alpha (values transparency), style (differentiate between catagories using point/line styling), markers (to view actual datapoints)
#     - scatterplot - size (control size of a point according to count of other var)
#     - histogram - can produce equal width with pd.qcut & equal height with below implementation 
#     - boxplot - sym="" (omit box outliers out of the calculation)
#     - relplot - col_wrap, col_order, ci="sd" for lineplots (get std information in plot)
#     - catplot - order (to control the order of categorical values)
#     - Joint plots - useful for comparing 2 numeric features
#      
# - Remember to call plt.show() at the end of a cell to view your visualization.  
# - We encourage you to save meaningful visualizations in the VISUALIZATIONS_PATH.
#

# + Collapsed="false"
# Example calls for all plots, ordered according to the above list.

# would put these examples near class EDAer

# sns.relplot(x='x_col_name', y='y_col_name', data=df, kind='line')
# sns.scatterplot('x_col_name', 'y_col_name', data=df, hue='color_col_name')
# df['col_name'].hist()
# sns.catplot(x='x_col_name', y='y_col_name', data=df, kind='bar')
# sns.catplot(x='x_col_name', y='y_col_name', data=df, kind='box')
# sns.catplot(x='x_col_name', y='y_col_name', data=df, kind='violin')
# sns.catplot(x='x_col_name', y='y_col_name', data=df, kind='point')
# sns.countplot(x='col_name', data=df, hue='color_col_name') # vertical
# sns.countplot(y='col_name', hue='color_col_name') # horizontal
# sns.heatmap(df) # be careful with this one...
# sns.relplot(x='x_col_name', y='y_col_name', data=df, kind='scatter', col='category_col_name_a', row='category_col_name_b')
# sns.catplot(x='x_col_name', y='y_col_name', data=df, kind='count', col='category_col_name_a', row='category_col_name_b')
# sns.jointplot(x='x_col_name', y='y_col_name', data=df)

# + Collapsed="false"
#export

# method in class EDAer
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

    
# method in class EDAer
def hist_equal_height(x, num_bins):
    """
    Example usage - plt.hist(df['numeric_col_name'], hist_equal_height(df['numeric_col_name'], num_bins))
    Function returns calculated values + Can visualize result with plt.show()
    """
    npt = len(x)
    return np.interp(np.linspace(0, npt, num_bins + 1),
                     np.arange(npt),
                     np.sort(x))

# + [markdown] Collapsed="false"
# We can also do this interactively! let's see how the features interact with the target variable

# + Collapsed="false" jupyter={"outputs_hidden": true}
#jupyter_code
# if possible without breaking non-jupyter apps, make these methods in class EDAer

# plotting_dict = {
#     'line': {'f': sns.relplot, 'kwargs': {'kind':'line'}},
#     'scatterplot': {'f': sns.scatterplot, 'kwargs': {}},
#     'bar': {'f': sns.catplot, 'kwargs': {'kind':'bar'}},
#     'box': {'f': sns.catplot, 'kwargs': {'kind':'box'}},
#     'violin': {'f': sns.catplot, 'kwargs': {'kind':'violin'}},
#     'point': {'f': sns.catplot, 'kwargs': {'kind':'point'}},
#     'countplot_vertical': {'f': sns.countplot, 'kwargs': {}},
#     'countplot_horizontal': {'f': sns.countplot, 'kwargs': {}},
# }


# kinds = list(plotting_dict.keys())

# import ipywidgets as widgets
# @widgets.interact(feature=list(df.columns), kind=kinds)
# def feature_interact_with_target(feature='Name_Len', kind='line'):
#     plotting_dict[kind]['f'](x=feature, y=TARGET, data=df, **plotting_dict[kind]['kwargs'])

# + [markdown] Collapsed="false"
# ### Exploratory Correlations
# Can be divided into:  
# - **Linear** correlation between the variables - Pearson
# - **Monotonous** correlation between the variables (AKA Ranked correlation) - Spearman
# - **Ordinal association** between the variables (variables are growing/decreasing together) - Kendall
#    
# It's a good idea to also check out the p-value to assess the validity of the correlation

# + Collapsed="false"
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

# method in class DataCentral
def change_if_binary(df, target, summary_table):
    if df[target].nunique() == 2:
        print('######################################################################################')
        print('###################################### IMPORTANT #####################################')
        print('It appears this is a binary problem. Automatically changing dtype of target to string')
        df, summary_table = change_dtype(df, summary_table, target, str)
        
    return df, summary_table
    


# + [markdown] Collapsed="false"
# ### Exploratory Clustering

# + [markdown] Collapsed="false"
# Occassionally, it will be important to identify clusters in our data.  
# Those clusters are usually based on:
# - **Numerical values proximity** - we'll use things like KMeans & GMMs to **cluster units** together
# - **Dendrogram** - Utilize some correlation metric to **cluster columns** together (e.g. through hierarchical clustering)
#    

# + Collapsed="false"
#export
# methods in class EDAer

def numerical_clustering(df, method='kmeans', num_clusters=4):
    """
    Currently only handles KMeans. TODOs:
    - refactor to create visualizations
    - add GMM
    - add sillouhette handling
    """
    if method == 'kmeans':
        clusters = KMeans(n_clusters=num_clusters).fit(df)
    
    return clusters
        



# + [markdown] Collapsed="false"
# #### Visualize Wafers

# + Collapsed="false"
#export
# methods in class EDAer
# TODOs - enable viewing multiple wafers, improve the wafer drawing
def get_all_possible_x_y_coordinates(df):
    """This method is used as part of drawing the wafer map,
    whose shape might change from product to product"""
    return df[['x','y']].drop_duplicates()

def is_coord_in_df(df,die_x,die_y):
    """Helper function to decide whether a dataFrame holds an indicated die"""
    return ((df['x'] == die_x) & (df['y'] == die_y)).any()


    


# + [markdown] Collapsed="false"
# ## Data Validation & Preprocessing

# + [markdown] Collapsed="false"
# The validation step will run an analysis based on our EDA & produce warning for our convenience regarding the data.  
# Moreover, in this section we centralize the location of ALL the data transformations we're doing to the input. It is comprised of:
# - Handling of missing values (omitting & filling information)
# - Handling Data Types
# - Normalization methodology (if necessary)
# - Feature Engineering

# + Collapsed="false"
#export
class DataCentral():
    """
    TODO finalize usage & implement across the notebook
    Main idea is that we'll build an object that will include all the needed information to preprocess our input data in a traceable manner,
    so that we'll be able to easily work with this single object.
    """
    
    def __init__(self, df, edaer=True):
        """
        TODO this should receive the path to the dataframe and load it itself to orig_df
        """
        self._orig_df = df
        self.df = df.copy()
        self.summary_table = SummaryTable(df)
        
        self.edaer = None
        if edaer:
            self.edaer = EDAer()

    def restart(self):
        self.df = self.orig_df.copy()

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
# ### EDA-based decisions & insights
#

# + Collapsed="false"
#export


# all of this logic should be under DataCentral, probably unified under a class
# we'll rename decisions_and_insights_based_on_eda to something like generic_decisions_and_insights
def drop_duplicate_rows(df):
    orig_n_rows = df.shape[0]
    df = df.drop_duplicates()
    num_dups = orig_n_rows - df.shape[0]
    if num_dups > 0:
        print(f"() Recognized {num_dups} duplicate rows. Dropping them.")
    
    return df, num_dups > 0
    
def drop_rows_with_missing_target_value(df, target):
    
    with_target_vals = df.loc[~df[target].isnull()]
    num_dropped = df.shape[0] - with_target_vals.shape[0]
    
    if num_dropped > 0:
        print(f"() Dropping {num_dropped} units where the target column is missing.")
        print("this is a default behavior, but notice that there are other approaches")
        print("that can be taken for this case, E.g. imputing those values and making sure")
        print("they're only used for training.")
        print("If interested, please contact developers to allocate time for this feature.\n")

    return with_target_vals, num_dropped > 0

def handle_missing_target_values(df, target):
    """ this function will serve as the future base for handling semi-supervised use-cases"""
    return drop_rows_with_missing_target_value(df, target)


def drop_unique_and_single_values(df, summary_table, insights=None):
    """
    TODO add checks for the words unique, uuid etc.
    """

    # handle entirely unique
    all_unique = summary_table.loc[(summary_table['num_unique_values'] == df.shape[0]) & (~summary_table['should_drop']), 'col_name'].tolist()
    if insights is None:
        insights = pd.DataFrame(columns=['col_name','insight','default_action'])
    
    for c in all_unique:
        if summary_table.loc[summary_table['col_name'] == c, 'suggested_type'].values[0] == 'cat':
            print('() Column', c, 'is a categorical that is entirely unique, hence will provide little value. Dropping it.')
            df, summary_table = drop_columns(df, summary_table, all_unique, 'entirely unique values')
            
        elif is_integer_dtype(summary_table.loc[summary_table['col_name'] == c, 'current_dtype'].values[0]):
            print('() Column', c, 'is an integer that is entirely unique, might be an identifier. Logged it to insights.')
            insights = insights.append(pd.DataFrame({'col_name': [c], 'insight': ['UNIQUE_INT'], 'default_action': ['drop']}))
    
    # handle single values
    only_one = summary_table.loc[(summary_table['num_unique_values'] == 1) & (~summary_table['should_drop']), 'col_name'].tolist()
    if only_one:
        print('() Columns', only_one, 'have only 1 value, hence are useless for a model. Dropping them.')
        df, summary_table = drop_columns(df, summary_table, only_one, 'only 1 value')
    
    return df, summary_table, insights

def act_on_unique_int(df, summary_table, insights):
    # drop unique int columns
    reason = 'UNIQUE_INT'
    unique_ints_to_drop = insights.loc[insights['insight'] == reason, 'col_name'].tolist()
    df, summary_table = drop_columns(df, summary_table, unique_ints_to_drop, reason)
    
    return df, summary_table


def insights_from_pandas_profiling(pandas_profile, insights=None):
    if pandas_profile is not None:
        profile_json = json.loads(pandas_profile.to_json()) 
        prf_insights = pd.DataFrame({'col_name': [i.split(' ')[-1] for i in profile_json['messages']], 'insight': [i.split(' ')[0] for i in profile_json['messages']]})
    
        if insights is None:
            insights = prf_insights
    
        else:
            insights = pd.concat([insights, prf_insights],ignore_index=True)
    
    return insights

def alert_on_threhsold(df, summary_table, insights, types, col_thresh, threshold,act='leave as is'):
    """
    TODO Refactor all threshold-based alerts to use this method
    """
    suspects = summary_table.loc[(summary_table['suggested_type'] == types) & (summary_table[col_thresh] > threshold), 'col_name'].tolist()
    
    if suspects:
        print(f'() We identified {types} columns that had {col_thresh} larger than {threshold}, and logged them in the insights object.\n')
        
        insights = pd.concat([insights, pd.DataFrame({'col_name': suspects, 'insight': [f'HIGH_{col_thresh.upper()}_{types.upper()}']*len(suspects), 'default_action': [act]*len(suspects)})])
    
    return insights

def alert_on_numeric_low_cardinality(df, summary_table, insights, threshold=0.01):
    only_num = summary_table.loc[summary_table['suggested_type'] == 'numeric']
    min_cardinality = int(threshold*df.shape[0])
    low_card = only_num.loc[summary_table['num_unique_values'] <= min_cardinality, 'col_name'].tolist()
    
    if low_card:
        print('() We identified numerical columns that had less unique values than ' +str(threshold*100)+ '% of the dataframe size, and logged them in the insights object.\n')
        
        insights = pd.concat([insights, pd.DataFrame({'col_name': low_card, 'insight': ['LOW_CARDINALITY_NUMERICAL']*len(low_card), 'default_action': ['leave as is']*len(low_card)})])
    
    return insights

def alert_on_categorical_high_cardinality(df, summary_table, insights, threshold=100):
    high_card = summary_table.loc[(summary_table['suggested_type'] == 'cat') & (summary_table['num_unique_values'] > threshold), 'col_name'].tolist()
    
    if high_card:
        print(f'() We identified categorical columns that had more unique values than {threshold} values, and logged them in the insights object.\n')
        
        insights = pd.concat([insights, pd.DataFrame({'col_name': high_card, 'insight': ['HIGH_CARDINALITY_CATEGORICAL']*len(high_card), 'default_action': ['drop']*len(high_card)})])
    
    return insights

def act_on_categorical_high_cardinality(df, summary_table, insights):
    # drop unique int columns
    reason = 'HIGH_CARDINALITY_CATEGORICAL'
    high_cards_to_drop = insights.loc[insights['insight'] == reason, 'col_name'].tolist()
    df, summary_table = drop_columns(df, summary_table, high_cards_to_drop, reason)
    
    return df, summary_table

def alert_on_high_std(df, summary_table, insights, threshold=1e6):
    return alert_on_threhsold(df, summary_table, insights, 'numeric', 'std', threshold,act='leave as is')

def alert_on_high_mean(df, summary_table, insights, threshold=1e6):
    return alert_on_threhsold(df, summary_table, insights, 'numeric', 'mean', threshold,act='leave as is')


def default_actions(df, summary_table, insights):
    df, summary_table = act_on_categorical_high_cardinality(df, summary_table, insights)
    df, summary_table = act_on_unique_int(df, summary_table, insights)
    return df, summary_table


def show_insights_summary(insights):
    return insights.groupby("insight").agg({'col_name': ['count','first'], 'default_action':['last']}).rename({'count':'# of columns', 'first':'first column', 'last': 'default action'},axis=1).droplevel(0,axis=1)


def show_insight_columns(proc_df, summary_table, insights, insight_type, amount=5):
    suspects = insights.loc[insights['insight'] == insight_type,'col_name'].to_list()
    suspects = summary_table.loc[(~summary_table['should_drop']) & (summary_table['col_name'].isin(suspects)), 'col_name'].to_list()

    if suspects:
        disp(proc_df[suspects].head(amount))
    else:
        print("no column has that type of insight")


def decisions_and_insights_based_on_eda(df, summary_table, target, pandas_profile=None, anlz_folder=None, update_summary=False, default_acts=False):
    print("--- producing decisions and insights based on eda phase ---\n")
    
    if update_summary:
        print('first, updating summary_table statistics')
        summary_table = update_summary_table_stats(df, summary_table)
    
    df, semi_supervised = handle_missing_target_values(df, target)

    df, dropped_dups = drop_duplicate_rows(df)
    
    if semi_supervised or dropped_dups: # we might have significantly changed the statistics - update the summary table accordingly
        summary_table = update_summary_table_stats(df, summary_table)
    
    df, summary_table, insights = drop_unique_and_single_values(df, summary_table)
    insights = alert_on_numeric_low_cardinality(df, summary_table, insights)
    insights = alert_on_categorical_high_cardinality(df, summary_table, insights)
    
    # if available, extract insights from profile
    insights = insights_from_pandas_profiling(pandas_profile, insights)
    
    print("--- finished producing decisions and insights based on eda phase ---")
    if not insights.empty:
        
        print("We were able to detect some interesting insights.")
        if anlz_folder:
            print("Saving them")
            with open(anlz_folder/'generic_insights.p', 'wb') as f:
                pickle.dump(insights, f)
    
        if default_acts:
            print("taking default actions for insights")
            df, summary_table = default_actions(df, summary_table, insights)
    
    return df, insights, summary_table




def drop_columns_with_missing_vals(proc_df,summary_table,thresh_fact):
    print(f"dropping columns with more than {100*thresh_fact}% of missing values")
    missing_cols_to_drop = summary_table[summary_table['missing_percent'] >= (100*thresh_fact)]['col_name'].tolist()
    print(f"There were {len(missing_cols_to_drop)} columns that are too sparse and will be dropped according to configuration:\n\n", missing_cols_to_drop[: min(len(missing_cols_to_drop), 10)])

    # update our dropped columns tracker
    reason = "missing values > " + str(thresh_fact)
    proc_df, summary_table = drop_columns(proc_df, summary_table, missing_cols_to_drop, reason)
    
    return proc_df, summary_table

def drop_rows_with_missing_vals(proc_df, summary_table, thresh_fact):
    print(f"dropping rows with more than {100*thresh_fact}% of missing values")
    drop_rows_rule = (proc_df.isnull().sum(axis=1)/len(proc_df.columns)) < thresh_fact
    proc_df = proc_df[drop_rows_rule]

    rows_dropped = drop_rows_rule.shape[0] - proc_df.shape[0]
    print("number of rows that are too sparse and will be dropped according to configuration:", rows_dropped)

    if rows_dropped:
        summary_table = update_summary_table_stats(proc_df, summary_table)
        
    return proc_df, summary_table

# + Collapsed="false"
#export
# method in DataCentral. Refactor to only receive specific needed params
def drop_percentage_logic(proc_df, summary_table, config, additional_cols_to_drop=None):
    """
    Convenience wrapper for handling missing values dropping logic
    """
    proc_df,summary_table = drop_columns_with_missing_vals(proc_df,summary_table,config['PREPROCESSING_CONFIG']['DROP_MISSING_COL_PERCENT'])
    proc_df, summary_table = drop_rows_with_missing_vals(proc_df, summary_table, config['PREPROCESSING_CONFIG']['DROP_MISSING_ROW_PERCENT'])
    
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
#export

# class DataImpute. Will be a property under DataCentral or a DataProcessor class, need to discuss
def drop_cat_cols_with_missing_vals(proc_df, summary_table):
    """
    TODO develop additional methods to handle categorical values. Among options:
    - for few %, throw rows (make sure they're not the "small" class)
    - series.mode()
    
    """
    RULE_FOR_CAT = (summary_table['missing_percent'] > 0) & (summary_table['suggested_type'] == 'cat') & (summary_table['current_dtype'] == 'object')
    cols_to_drop = summary_table.loc[RULE_FOR_CAT, 'col_name'].tolist()
    if cols_to_drop:
        print(f"there were {len(cols_to_drop)} categorical columns with missing values. Dropping them as default")
        reason = "missing categorical values"
        proc_df, summary_table = drop_columns(proc_df, summary_table, cols_to_drop, reason)
    
    return proc_df, summary_table

def data_impute(data, impute_features, impute_groups = [], impute_func = 'median'):
    """
    This method replaces the missing values in each column in the impute_features, grouped by the impute_groups
    by defualt the values replaced by the median of the column (per group)
    
    inputs: data frame, list of features to impute, list of groups
    Returns: data frame with imputed values
    """
    
    if len(impute_groups) == 0:
        if impute_func == 'median':
            output_data = data[impute_features].transform(lambda x: x.fillna(x.median()))        
        elif impute_func == 'mean':
            output_data = data[impute_features].transform(lambda x: x.fillna(x.mean()))
    else:
        
        if impute_func == 'median':
            output_data = data.groupby(impute_groups)[impute_features].transform(lambda x: x.fillna(x.median()))        
        elif impute_func == 'mean':
            output_data = data.groupby(impute_groups)[impute_features].transform(lambda x: x.fillna(x.mean()))
    
    return output_data

def data_impute_numeric(proc_df, summary_table, impute_groups = None, impute_func = None, config = None):
    ########### SUBTLE DETAIL ###########
    # we're using the suggested type rather than the actual dtype since those are still flawed by the NaNs
    
    if config is not None:
        impute_groups = config['PREPROCESSING_CONFIG']['IMPUTE_GROUPS']
        impute_func = config['PREPROCESSING_CONFIG']['IMPUTE_FUNC']
    
    impute_features = summary_table.loc[(summary_table['suggested_type'] == 'numeric') & (~summary_table['should_drop']), 'col_name'].tolist()
    proc_df[impute_features] = data_impute(proc_df, impute_features, impute_groups, impute_func)

    return proc_df



# + [markdown] Collapsed="false"
# ### Handle Data Types (Numeric, Categorical, Datetime)
# Often, the default data-type assigned to a column is not the correct type.  
# This can hurt us in several ways. Some examples:
# 1. **wasting computation resources** - e.g. numeric as float64 when it can be uint8
# 2. **missing sub-categorization** - e.g. reading datetime as an object, missing out on datetime transformations
# 3. **just plain wrong behavior** - e.g. numeric representation for a non-ordinal category, like reading "x" coordinate on a wafer as a numeric   
#
# In our summary object, we've already used some heuristics to try & figure out what class the column should belong to.  
# For seeing the current mapping, we can look at these properties of the summary table
#

# + Collapsed="false"
# Maybe change some more
# proc_df, summary_table = change_dtype(proc_df, summary_table, 'meta_HardBin_EWST1', 'category')

# + [markdown] Collapsed="false"
# By now we should be confident in our data types. Let's wrap this process in a function

# + Collapsed="false"
#export
# methods under DataCentral
def view_df_with_dtypes(df):
    col_and_dtype = {x: x + " ("+str(df[x].dtype)+")" for x in df.columns}
    return df.rename(mapper=col_and_dtype,axis=1)

def change_dtypes_by_summary_table(proc_df, summary_table):
    type2cols_dict = get_dtype_categories_from_summary(summary_table)

    # change booleans
    for c in type2cols_dict['numeric']:
        if proc_df[c].dtype == 'bool':
            proc_df, summary_table = change_dtype(proc_df, summary_table, c, int)
    
    # change categoricals
    for c in type2cols_dict['cat']:
        proc_df[c] = pd.Categorical(proc_df[c])

    # try changing datetime
    failed_to_datetime = []
    for c in type2cols_dict['time']:
        try:
            proc_df[c] = pd.to_datetime(proc_df[c], infer_datetime_format=True)
        except Exception:
            failed_to_datetime.append(c)

    if failed_to_datetime:
        print("we failed to change some columns to datetime. Will default to dropping them")
        print("The columns that failed:")
        print(failed_to_datetime)
        
    return proc_df, failed_to_datetime

def handle_datatypes_logic(proc_df, summary_table):
    """
    Will change all columns into their suggested type. 
    Notice that the generic behavior of failing to turn into datetime is dropping the column!
    """
    
    proc_df, failed_to_datetime = change_dtypes_by_summary_table(proc_df, summary_table)
    
    # generic behavior is dropping these columns. 
    failed_to_datetime_drop = failed_to_datetime
    failed_to_datetime_change_to_cat = [] # TODO should drop or involve in logic

    proc_df, summary_table = drop_columns(proc_df, summary_table, failed_to_datetime_drop, "failed to change to datetime")

    for c in failed_to_datetime_change_to_cat:
        proc_df[c] = pd.Categorical(proc_df[c])

        # We'll also update our summary understanding
        summary_table.loc[summary_table['col_name'] == c, 'suggested_type'] = 'cat'
    
    
    return proc_df, summary_table


# + [markdown] Collapsed="false" toc-hr-collapsed=true toc-nb-collapsed=true Collapsed="false"
# ### Datetime - Missing Values & Column Enrichment

# + Collapsed="false"
#export



# + [markdown] Collapsed="false"
# ### Assess Temporal Influence
# See [User Story](https://ni.visualstudio.com/DevCentral/_backlogs/backlog/OptimalPlus%20Data%20Science/Initiatives/?team=OptimalPlus%20Data%20Science&workitem=1152436)  
# Basically use time as splitMethod + run the compare populations pipeline.  
# another simpler concept but useful is to utilize pandas series properties for is_monotonic_decreasing, is_monotonic_increasing

# + [markdown] Collapsed="false"
# ### Normalizations

# + Collapsed="false"

#export

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


# + [markdown] Collapsed="false" toc-hr-collapsed=true toc-nb-collapsed=true Collapsed="false"
# ## Feature Engineering - Under Construction, skip for now

# + [markdown] Collapsed="false"
# Let's produce some features based on those.  
# In order to utilize those flexibly, for new projects, we'll implement those using an agreed-upon interface - all engineered features will be produced using functions in this structure:
# - input parameters - a dataframe + additional k,v pairs with indicative naming
# - returns a dataframe augmented with the engineered features as additional columns
#
# Notes:
# - In case some features should be dropped after this operation, add those to summary
#

# + Collapsed="false"
#export


# + [markdown] Collapsed="false"
# And as always, wrap those as a utility

# + Collapsed="false"
#export



# TODO add to configurations.
# set_background('lightyellow')
# GLOBAL_CONFIG['PREPROCESSING_CONFIG']['FE_FUNCS'] = [polynomial_features]
# proc_df = feature_engineering(proc_df, GLOBAL_CONFIG['PREPROCESSING_CONFIG']['FE_FUNCS'])


# + [markdown] Collapsed="false"
# ## Data Splitting

# + [markdown] Collapsed="false"
# For classification - If our data is highly imbalanced, it might affect our data-sampling & splitting strategies.  
# For Regression, if the values range is very large, it might be worthwhile to log that range (TODO find justification, should be especially important for parameterized models

# + Collapsed="false"
#export

# methods under DataCentral. We'll have a "prepare_for_training" method that does all the preprocessing & splitting



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
### Configuration cell ###
# set_background('lightyellow')


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
        proc_df, insights, summary_table = decisions_and_insights_based_on_eda(df, summary_table, config['TARGET'])
    
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