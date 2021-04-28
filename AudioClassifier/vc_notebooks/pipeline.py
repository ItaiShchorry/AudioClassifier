# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: audio_cls
#     language: python
#     name: audio_cls
# ---

# + [markdown] Collapsed="false"
# # Audio Classification

# + [markdown] Collapsed="false"
# ## Resources

# + [markdown] Collapsed="false"
# The project builds upon knowledge, practices & code gathered from the following list of resources:
# - Full Stack Deep Learning materials
# - Made With ML "MLOps" series - https://madewithml.com/courses/mlops/
# - Mikes Males's Sound Classification project - https://mikesmales.medium.com/sound-classification-using-deep-learning-8bc2aa1990b7
# - Fast.ai's course & library

# + [markdown] Collapsed="false"
# ## Objective

# + [markdown] Collapsed="false"
# The main objective of this project is to experience a high-standard of E2E ML project development, from as many aspects as possible given time constraints.   
# Admittedly, the problem itself was chosen to allow flexibility & diversity in the approach taken (can be treated as a sequence, an image and as a tabular problem),  
# while choosing a problem that might still have room to improve results upon (benchmarks are around 92%).  
# Hopefully those traits will force the work to be relatively generic, and serve as building block for future work.
#
#

# + [markdown] Collapsed="false"
# # Initial Imports, Methods & Classes

# + Collapsed="false"
# temp cell until we'll standardize usage of library

# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import sys
cwd = Path.cwd()
path_library = str(cwd.parent/'vc_notebooks')
sys.path.append(path_library)

from data import *

# + Collapsed="false"
# visualize a controllable size of the datafame
from IPython.display import display,HTML
def disp(df,max_rows=30, max_cols=1000):
    with pd.option_context("display.max_rows", max_rows):
        with pd.option_context("display.max_columns", max_cols):
            display(df)

def grab_cell_vars(offset=0):
    """
    Used to make a config out of defined flags in cell
    """
    import io
    from contextlib import redirect_stdout

    ipy = get_ipython()
    out = io.StringIO()

    with redirect_stdout(out):
        ipy.magic("history {0}".format(ipy.execution_count - offset))

    #process each line...
    x = out.getvalue().replace(" ", "").split("\n")
    x = [a.split("=")[0] for a in x if "=" in a] #all of the variables in the cell
    g = globals()
    result = {k:g[k] for k in x if k in g}
    return result

# Cell coloring code. 
def set_background(color):    
    script = (
        "var cell = this.closest('.jp-CodeCell');"
        "var editor = cell.querySelector('.jp-Editor');"
        "editor.style.background='{}';"
        "this.parentNode.removeChild(this)"
    ).format(color)

    display(HTML('<img src onerror="{}">'.format(script)))
    
from IPython.core.magic import register_cell_magic


@register_cell_magic
def background(color, cell):
    set_background(color)


# + [markdown] Collapsed="false"
# # Data

# + [markdown] Collapsed="false"
# We'll start by initializing a config which will be used to (hopefully) make our decisions throughout the analysis, preprocessing & model building fully traceable

# + Collapsed="false"
# Create config to include all our different decision making. 

GLOBAL_CONFIG = {}    
BASE_FOLDER = Path(r'/dsp/dsp_portal/personal/itai.shchorry/AudioClassificationProject/AudioClassifier/Data/preprocessed/mfcc') # Path containing your data. 
CONFIG_PATH = BASE_FOLDER/'config.p'

GLOBAL_CONFIG['BASE_FOLDER'] = BASE_FOLDER
GLOBAL_CONFIG['CONFIG_PATH'] = CONFIG_PATH


# + Collapsed="false"
def update_and_persist(config, added_dict, path=CONFIG_PATH):
    """
    Method to help maintain the state of the configuration file
    """
    config.update(added_dict)
    with open(CONFIG_PATH,'wb') as f:
        pickle.dump(config, f)


# + [markdown] Collapsed="false"
# Let's set some initial configurations
#

# + Collapsed="false"
### Configuration cell ###
set_background('lightyellow')

FILE_PATH = str(BASE_FOLDER/'data.p')
TARGET = 'y' 

#################### DO NOT TOUCH ####################
if "DATA_CONFIG" in locals():
    del DATA_CONFIG

ANALYSIS_RESULT_FOLDER = BASE_FOLDER/(Path(FILE_PATH).stem) # This folder will contain all analysis outputs from our work
DATA_CONFIG = grab_cell_vars()

# make paths serializable
DATA_CONFIG['BASE_FOLDER'] = str(BASE_FOLDER)
DATA_CONFIG['ANALYSIS_RESULT_FOLDER'] = str(ANALYSIS_RESULT_FOLDER)

update_and_persist(GLOBAL_CONFIG,DATA_CONFIG)

# + [markdown] Collapsed="false"
# We'll start by loading the data to memory.  

# + Collapsed="false"
df = read_df_from_path(GLOBAL_CONFIG['FILE_PATH'])

# + [markdown] Collapsed="false"
# ## Generic Exploration Data Analysis

# + [markdown] Collapsed="false"
# Let's look at the first few rows our dataframe.  
#

# + Collapsed="false"
disp(df, max_rows=5, max_cols=1000) 

# + [markdown] Collapsed="false"
# ### Create Summary View & Profiler
# We'll summarize some meta-level information + aggregative measures for each column

# + Collapsed="false"
summary_table = create_col_level_summary_table(df)
disp(summary_table,max_rows=41)

# + [markdown] Collapsed="false"
# This summary can already give us some insights on the data and actions that we'd like to take, e.g. drill-down a high-cardinality column or drop one with many missing values.   
# Those will be part of the next section, which handles data validation & data transformations as a preparation for running the model.  
#

# + [markdown] Collapsed="false"
# ### Exploratory Visualizations

# + [markdown] Collapsed="false"
# Let's expose a quick way to visualize interactions with the target

# + Collapsed="false"
plotting_dict = {
    'line': {'f': sns.relplot, 'kwargs': {'kind':'line'}},
    'scatterplot': {'f': sns.scatterplot, 'kwargs': {}},
    'bar': {'f': sns.catplot, 'kwargs': {'kind':'bar'}},
    'box': {'f': sns.catplot, 'kwargs': {'kind':'box'}},
    'violin': {'f': sns.catplot, 'kwargs': {'kind':'violin'}},
    'point': {'f': sns.catplot, 'kwargs': {'kind':'point'}},
    'countplot_vertical': {'f': sns.countplot, 'kwargs': {}},
    'countplot_horizontal': {'f': sns.countplot, 'kwargs': {}},
}


kinds = list(plotting_dict.keys())

# + Collapsed="false"
import ipywidgets as widgets

@widgets.interact(feature=list(df.columns), kind=kinds)
def feature_interact_with_target(feature='mfcc_2', kind='box'):
    plotting_dict[kind]['f'](x=feature, y=TARGET, data=df, **plotting_dict[kind]['kwargs'])


# + [markdown] Collapsed="false"
# It seems like there's some differences in the features for each class, increasing the confidence in the ability to classify according to them.

# + [markdown] Collapsed="false"
# ### Exploratory Correlations
# Can be divided into:  
# - **Linear** correlation between the variables - Pearson
# - **Monotonous** correlation between the variables (AKA Ranked correlation) - Spearman
# - **Ordinal association** between the variables (variables are growing/decreasing together) - Kendall
#    
# Can change method parameter to the relevant connection

# + [markdown] Collapsed="false"
# Let's calculate all possible pair-wise correlations for our DataFrame (besides columns which have more than 50% missing values).  
#

# + Collapsed="false"
corr_table = df.corr(method='spearman',min_periods=int(df.shape[0]*0.5))
heatmap = sns.heatmap(corr_table)


# + [markdown] Collapsed="false"
# Another visualization for the correlations will be through hierarchical clustering

# + Collapsed="false"
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
flat_corr_table = scipy.cluster.hierarchy.distance.squareform((1-corr_table))
hac = linkage(flat_corr_table, optimal_ordering=True)

fig = plt.figure(figsize=(10,6))
dendrogram(hac, labels=corr_table.columns)
plt.show()

# + [markdown] Collapsed="false"
# Doesn't seem very interesting to continue pursuing at the moment (makes sense as we know how these features were generated)

# + [markdown] Collapsed="false"
# ### Target Exploration

# + [markdown] Collapsed="false"
# What are some raw statistics for our target?

# + Collapsed="false"
print("The target is of type:", df[TARGET].dtype)

target_type = 'category' if str(df[TARGET].dtype) in ['category', 'object'] else 'numeric'

#descriptive statistics summary
disp(summary_table[summary_table['col_name'] == TARGET])


# + [markdown] Collapsed="false"
# Let's plot its distribution

# + Collapsed="false"
fig, ax = plt.subplots(figsize=(14,6))
sns.countplot(x=TARGET, data=df,ax=ax)



# + [markdown] Collapsed="false"
# Car horn and gun shot classes are under-represented.  
# We'll probably want to address that later on.

# + [markdown] Collapsed="false"
# ## Data Validation & Preprocessing

# + [markdown] Collapsed="false"
# The validation step will run an analysis based on our EDA & produce warning for our convenience regarding the data.  
# Moreover, in this section we centralize the location of ALL the data transformations we're doing to the input. It is comprised of:
# - Handling Data Types
# - Normalization methodology (if necessary)
#
# **Note** - We won't need to deal with missing values & imputations as the features were extracted - we have a complete dataframe.

# + [markdown] Collapsed="false"
# As always, we'll start by updating the configuration.

# + Collapsed="false"
### Configuration cell ###
set_background('lightyellow')

# outliers
REMOVE_OUTLIERS = False
OUTLIER_SIGMA_CUTOFF = 5

# Normalization handling
PERFORM_NORMALIZATION = True
NORMALIZATION_METHOD  =  'minmax' # choose from: minmax, zscore
NORMALIZATION_MAPPING = {1: {"groups":[], "columns":[]}}

# Set random seed for reproducibility
RANDOM_STATE = 42 

#################### DO NOT TOUCH ####################

# set random seed
set_seeds(RANDOM_STATE)

# Make sure we don't constantly multiply the config
if "PREPROCESSING_CONFIG" in locals():
    del PREPROCESSING_CONFIG
    
PREPROCESSING_CONFIG = grab_cell_vars()
update_and_persist(GLOBAL_CONFIG,PREPROCESSING_CONFIG)



# + [markdown] Collapsed="false"
# In case we're unhappy with our preprocessing, we can come back to this cell and start over

# + Collapsed="false"
proc_df = df.copy()



# + [markdown] Collapsed="false"
# ### Outlier detection

# + [markdown] Collapsed="false"
# Outliers tend to have a bad effect on training - let's see if we recognize some.  
# We'll take a simple approach of dropping according to univariate sigma-cutoff analysis.

# + Collapsed="false"
if REMOVE_OUTLIERS:
    outlier_df = proc_df.copy().drop(TARGET,axis=1)

    for c in proc_df.drop(TARGET,axis=1).columns:
        mean = summary_table.loc[summary_table['col_name']== c,'mean'].values[0]
        std = summary_table.loc[summary_table['col_name']== c,'std'].values[0]

        cut_off = std * OUTLIER_SIGMA_CUTOFF
        lower, upper = mean - cut_off, mean + cut_off
        outlier_df[c] = ~((lower <= outlier_df[c]) & (outlier_df[c] <= upper ))


    outliers = outlier_df[outlier_df.any(axis=1)].index.to_list()
    proc_df = proc_df.loc[~proc_df.index.isin(outliers)]
    print(f"dropped {len(outliers)} outliers")

# + [markdown] Collapsed="false"
# Let's hear if their sound is indeed different compared to other inputs in the class

# + Collapsed="false"
import ipywidgets as widgets
import IPython.display as ipd
import librosa
import librosa.display
AUDIO_FOLDER = Path(r'/dsp/dsp_portal/personal/itai.shchorry/AudioClassificationProject/AudioClassifier/Data/UrbanSound8K/audio')

if REMOVE_OUTLIERS and outliers:
    outlier_cls = df.loc[outlier, TARGET]
    filename = AUDIO_FOLDER/outlier
    plt.figure(figsize=(12,4))
    data,sample_rate = librosa.load(filename)
    print(f"class: {outlier_cls}")
    _ = librosa.display.waveplot(data,sr=sample_rate)
    ipd.Audio(filename)

# + Collapsed="false"
if REMOVE_OUTLIERS and outliers:
    same_cls = proc_df[proc_df[TARGET] == outlier_cls].index[0]
    filename = AUDIO_FOLDER/same_cls
    plt.figure(figsize=(12,4))
    data,sample_rate = librosa.load(filename)
    print(f"class: {outlier_cls}")
    _ = librosa.display.waveplot(data,sr=sample_rate)
    ipd.Audio(filename)

# + [markdown] Collapsed="false"
# From hearing several examples, it looks like these recording instances are indeed confusing.  
# A plausible approach would be to discard those from training & apply the outlier detector as a safety mechanism during inference, but for the purpose of this project we'll prefer investing in exercising the course's materials.  
# In case of future regret, we'll wrap this logic in a function in the data module (called drop_outliers)

# + [markdown] Collapsed="false"
# ### Normalizations

# + Collapsed="false"
# Prepare normalization mapping 
proc_df = normalize(proc_df, GLOBAL_CONFIG, summary_table=summary_table)

# + [markdown] Collapsed="false"
# # Model

# + [markdown] Collapsed="false"
# We're done understanding & preprocessing our data, we can start training some models.  
# "Simple" tabular models will be either rather shallow fully-connected-based models, or, more commonly, tree-based methods, which often even triumph DL methods on tabular data.  
# So, we'll start with several of those as baselines, move forward to fast.ai's battle-tested implementation for tabular data, validate we're able to overfit a single batch, and continue according to the bias-variance decomposition. 
#

# + Collapsed="false"
from model import *

# + [markdown] Collapsed="false"
# ## Model Configurations

# + [markdown] Collapsed="false"
# We begin by setting configurations

# + Collapsed="false"
### Configuration cell ###
# set_background('lightyellow')

BATCH_SIZE = 64

TRAIN_SIZE = 0.7
VALID_SIZE = 0.15
TEST_SIZE = 0.15


# #################### DO NOT TOUCH ####################
SUMMARY_TABLE_PATH = str(BASE_FOLDER/'summary_table.p')

if "MODEL_CONFIG" in locals():
    del MODEL_CONFIG

MODEL_CONFIG = grab_cell_vars()
update_and_persist(GLOBAL_CONFIG,MODEL_CONFIG)

# log decisions of summary table
with open(SUMMARY_TABLE_PATH,'wb') as f:
    pickle.dump(summary_table, f)


# + Collapsed="false"
from fastai.tabular.all import *


# + [markdown] Collapsed="false"
# ## Data Splitting

# + [markdown] Collapsed="false"
# We'll want to perform stratified splitting into train, validation and test, so we can later apply iterations of bias-variance decomposition.  
#

# + Collapsed="false"
X_train, X_, y_train, y_ = train_test_split(proc_df.drop(TARGET, axis=1),  proc_df[TARGET], train_size=TRAIN_SIZE, random_state=RANDOM_STATE, stratify=proc_df[TARGET])

# + Collapsed="false"
print (f"train: {len(X_train)} ({(len(X_train) / len(proc_df)):.2f})\n"
       f"remaining: {len(X_)} ({(len(X_) / len(proc_df)):.2f})")

# + Collapsed="false"
# Split to test
X_valid, X_test, y_valid, y_test = train_test_split(X_, y_, train_size=round(VALID_SIZE/(VALID_SIZE+TEST_SIZE), 2), stratify=y_)

# + [markdown] Collapsed="false"
# Let's validate that the results look reasonable

# + Collapsed="false"
print(f"train: {len(X_train)} ({len(X_train)/len(proc_df):.2f})\n"
      f"val: {len(X_valid)} ({len(X_valid)/len(proc_df):.2f})\n"
      f"test: {len(X_test)} ({len(X_test)/len(proc_df):.2f})")

# + Collapsed="false"
y_valid.value_counts() / proc_df[TARGET].value_counts()


# + [markdown] Collapsed="false"
# Looks good. Let's save them for reproducibility & get those into fast.ai's dataloaders

# + Collapsed="false"
def pickles(p,x):
    with open(p,'wb') as f:
        pickle.dump(x, f)
        
# pickles(PREPROCESSED_FOLDER/'X_train.p', X_train)
# pickles(PREPROCESSED_FOLDER/'X_valid.p', X_valid)
# pickles(PREPROCESSED_FOLDER/'y_train.p', y_train)
# pickles(PREPROCESSED_FOLDER/'y_valid.p', y_valid)
# pickles(PREPROCESSED_FOLDER/'X_test.p', X_test)
# pickles(PREPROCESSED_FOLDER/'y_test.p', y_test)


# + Collapsed="false"
def picklel(p):
    with open(p,'rb') as f:
        x = pickle.load(f)
    
    return x
        
# X_train = picklel(PREPROCESSED_FOLDER/'X_train.p')
# X_valid = picklel(PREPROCESSED_FOLDER/'X_valid.p')
# y_train = picklel(PREPROCESSED_FOLDER/'y_train.p')
# y_valid = picklel(PREPROCESSED_FOLDER/'y_valid.p')
# X_test = picklel(PREPROCESSED_FOLDER/'X_test.p')
# y_test = picklel(PREPROCESSED_FOLDER/'y_test.p')


# + Collapsed="false"
y_train.value_counts() / proc_df[TARGET].value_counts()

# + [markdown] Collapsed="false"
# and wrap up entire process as a splitting method (located in data.py)

# + [markdown] Collapsed="false"
# ## Compare Train & Validation Data Distributions

# + [markdown] Collapsed="false"
# we're finally going to train a model, but still not for our original cause.   
#
# We've just splitted the data, assuming that the train & validation data represent the same distribution, i.e. "the real distribution" of our data.   
# Maybe that assumption is false? perhaps there's some feature that clearly distinguishes between the populations that we didn't take into account?  
# To answer that question, we'll train a model **to predict whether a unit belongs to the training data or the validation data**.  
#
# For features that were identified as 'leaky' by our model, we'll run a [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test), a statistical operation to compare distributions.

# + Collapsed="false"
compare_train_val_df = compare_train_valid_populations(X_train.copy(), X_valid.copy(), rand_state=RANDOM_STATE)
compare_train_val_df.head()

# + [markdown] Collapsed="false"
# ### Overfit a single batch

# + [markdown] Collapsed="false"
# Let's validate that fast.ai's model is able to overfit to a single batch

# + Collapsed="false"
set_seed(RANDOM_STATE)

single_batch_ds = proc_df.sample(BATCH_SIZE*2)
splits = RandomSplitter(valid_pct=0.5)(range_of(single_batch_ds))
dls_single = TabularPandas(single_batch_ds, cont_names=list(set(single_batch_ds.columns) - set([TARGET])), y_names=TARGET,  procs=[], splits=splits).dataloaders(bs=BATCH_SIZE)


# + Collapsed="false"
learn_single = tabular_learner(dls_single, metrics=accuracy)

# + Collapsed="false" jupyter={"outputs_hidden": true}
learn_single.fit_one_cycle(1000)


# + Collapsed="false"
train_losses = learn_single.get_preds(ds_idx=0,with_loss=True)[2]
train_losses.mean()

# + [markdown] Collapsed="false"
# We're able to keep improving our score, so the model's implementation looks OK from that aspect

# + [markdown] Collapsed="false"
# ## Train Model

# + [markdown] Collapsed="false"
# Construct data loaders from our data

# + Collapsed="false"
Xy, y_temp = pd.concat([X_train,X_valid]), pd.concat([y_train,y_valid])
Xy[TARGET] = y_temp
splits = [range_of(X_train.shape[0]), list(range(X_train.shape[0], Xy.shape[0],1))]
proc_df_wrapped = TabularPandas(Xy, cont_names=list(set(Xy.columns) - set([TARGET])), y_names=TARGET, splits=splits, procs=[])
dls = proc_df_wrapped.dataloaders(bs=64)

# + Collapsed="false"
learn = tabular_learner(dls, metrics=accuracy,cbs=[SaveModelCallback,ReduceLROnPlateau])

# + Collapsed="false"
learn.fit_one_cycle(27)


# + [markdown] Collapsed="false"
# # Analysis

# + [markdown] Collapsed="false"
# Before trying to improve the model, let's dive into the current status of the algorithm

# + [markdown] Collapsed="false"
# ## Predictions Drill Down

# + Collapsed="false"
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# + Collapsed="false"
interp.most_confused()[:5]

# + [markdown] Collapsed="false"
# Looks pretty good. Main confusion points are between (children playing, dog barking, street music) and (jackhammer, drilling), and at least the 2nd group makes a lot of sense.  
# Also street music seems like an ambiguous class for the learner.  
# We might want to visit these specific examples later on.

# + [markdown] Collapsed="false"
# ## Tree Visualization

# + [markdown] Collapsed="false"
# Let's visualize a simple Decision Tree to understand whether there are some indicative decision boundaries in regards to the label.  
#

# + Collapsed="false"
draw_tree(X_train, pd.Series(pd.Categorical(y_train)), type_tree='multiclass')

# + [markdown] Collapsed="false"
# Not much to learn, but it seems like there are indeed areas more fitting for specific classes (e.g. that 3 -> 6 split that leads to class 8, the siren). 

# + [markdown] Collapsed="false"
# # Continued Iterative Development

# + [markdown] Collapsed="false"
# This section intends to use the bias-variance decomposition we learned in class 6.  
# For that, we first need to evaluate results on the test set

# + Collapsed="false"
from sklearn.metrics import accuracy_score

# + Collapsed="false"
test_df = X_test.copy()
test_dl = learn.dls.test_dl(test_df)

preds = learn.get_preds(dl=test_dl, with_decoded=True)


# + [markdown] Collapsed="false"
# We'll want to encode y so we can calculate the accuracy

# + Collapsed="false"
categorizer = learn.dls.fs[0]
y_test_encoded = y_test.apply(lambda x: int(categorizer.encodes(x)))

# + Collapsed="false"
test_acc = metrics.accuracy_score(y_test_encoded,  preds[2].numpy())
test_acc


# + [markdown] Collapsed="false"
# Let's produce a waterfall chart for all 3 populations

# + Collapsed="false"
def acc_on_pop(X_pop, y_pop):
    """
    Using existing model obtained in learner, we calculate accuracy for a given population
    """
    
    # get predictions
    pop_df = X_pop.copy()
    pop_dl = learn.dls.test_dl(pop_df)

    preds = learn.get_preds(dl=pop_dl, with_decoded=True)
    
    categorizer = learn.dls.fs[0]
    y_test_encoded = y_pop.apply(lambda x: int(categorizer.encodes(x)))
    
    pop_acc = metrics.accuracy_score(y_test_encoded,  preds[2].numpy())
    return pop_acc

def calc_accs():
    tr_acc = acc_on_pop(X_train, y_train)
    val_acc = acc_on_pop(X_valid, y_valid)
    test_acc = acc_on_pop(X_test, y_test)
    
    return tr_acc,val_acc,test_acc


# + Collapsed="false"
tr_acc,val_acc,test_acc = calc_accs()

# + Collapsed="false"
import waterfall_chart
a = ['goal-perf','train','valid','test']
b = [0.95,tr_acc-0.95,val_acc-tr_acc,test_acc-val_acc]
waterfall_chart.plot(a, b, formatting='{:,.3f}');

# + [markdown] Collapsed="false"
# Looks like we're currently overfitting. Going over possible approaches to handle it:
# - data augmentations
# - regularization
# - tuning hyperparameters
#
# We'll dive into the last 2 items as they're more readily available.  
# We'll also experiment with general performance boosters such as monte-carlo dropout
#

# + [markdown] Collapsed="false"
# ## Handling Overfitting

# + [markdown] Collapsed="false"
# ### Regularization

# + Collapsed="false"
learn = tabular_learner(dls, metrics=accuracy,cbs=[SaveModelCallback,ReduceLROnPlateau], wd=0.1)

# + Collapsed="false"
learn.fit_one_cycle(30)


# + [markdown] Collapsed="false"
# Looks better! let's take a closer look

# + Collapsed="false"
def get_wf_chart():
    tr_acc,val_acc,test_acc = calc_accs()
    print(f"accuracies are: train: {tr_acc}, valid: {val_acc}, test: {test_acc}")
    a = ['goal-perf','train','valid','test']
    b = [0.95,tr_acc-0.95,val_acc-tr_acc,test_acc-val_acc]
    waterfall_chart.plot(a, b, formatting='{:,.3f}');
    
get_wf_chart()

# + [markdown] Collapsed="false"
# Got slightly better - let's try tuning some hyper parameters by setting up W&B

# + [markdown] Collapsed="false"
# ### Experiment Tracking With W&B

# + Collapsed="false" jupyter={"outputs_hidden": true}
import wandb
wandb.init()

# + Collapsed="false"
from fastai.callback.wandb import *
learn = tabular_learner(dls, metrics=accuracy,cbs=[SaveModelCallback,ReduceLROnPlateau, WandbCallback()], model_dir='tuning', wd=0.1)

# + Collapsed="false"
learn.fit_one_cycle(30)

# + [markdown] Collapsed="false"
# ## Package training & hyperparameter optimization

# + [markdown] Collapsed="false"
# We'll want to evaluate some hyper parameters.  
# We would use W&B's Sweep methodology, but it only takes into account the model hyperparameters - there are some preprocessing hyperparameters that might affect the results of our solution, so we will go with something more manual which can later be extended & optimized

# + Collapsed="false"
import itertools

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

# + Collapsed="false"
import wandb
from fastai.callback.wandb import *

def create_dataloaders(X_train,X_valid,y_train,y_valid, target):
    Xy, y_temp = pd.concat([X_train,X_valid]), pd.concat([y_train,y_valid])
    Xy[target] = y_temp
    splits = [range_of(X_train.shape[0]), list(range(X_train.shape[0], Xy.shape[0],1))]
    proc_df_wrapped = TabularPandas(Xy, cont_names=list(set(Xy.columns) - set([target])), y_names=target, splits=splits, procs=[])
    dls = proc_df_wrapped.dataloaders(bs=64)
    
    return dls

def run_training_run(dls, hp, run_name):
    wandb.init(project='Audio Classification', name=run_name)
    learn = tabular_learner(dls, metrics=accuracy,cbs=[SaveModelCallback,ReduceLROnPlateau, WandbCallback()], model_dir=run_name, **hp)
    learn.fit_one_cycle(40)
    



# + [markdown] Collapsed="false"
# Let's grid search our hyper parameters

# + Collapsed="false" jupyter={"outputs_hidden": true}
for dh in data_hparams:
    run_config = GLOBAL_CONFIG.copy()
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



# + [markdown] Collapsed="false"
# Reviewing results in the W&B's UI, we managed to get 94.43% accuracy, 2% above our benchmark!  
#
# Let's take a closer look at that model

# + Collapsed="false"
mdl_name = 'tuning_cutoff_0_16'
perm_idx = int(mdl_name.split('_')[-1])
permute_dict = permutations_dicts[perm_idx]

learn = tabular_learner(dls, metrics=accuracy,cbs=[], model_dir=mdl_name, **permute_dict)
learn.load(Path.cwd()/mdl_name/'model')

# + Collapsed="false"
get_wf_chart()

# + [markdown] Collapsed="false"
# and review the confusion matrix

# + Collapsed="false"
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# + Collapsed="false"
interp.most_confused()[:5]

# + [markdown] Collapsed="false"
# Let's wrap up the hp search for future use (run_hp_search in model.py), and produce a utility for inferencing a .wav file.  
#

# + Collapsed="false"
import librosa
def extract_mfcc_features(filepath):
    try:
        audio, sample_rate = librosa.load(filepath, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", filepath)
        return None 
     
    return mfccsscaled

def from_audio_to_table(wav_file_path):
    data = extract_mfcc_features(wav_file_path)
    idx = wav_file_path.parent.name+'/'+wav_file_path.name

    # Convert into a Panda dataframe 
    X = pd.DataFrame([data], columns=['mfcc_'+str(i) for i in range(data.shape[0])],index=[idx])

    return X

def predict_wav(wav_file_path, learn, config):
    wave_df = from_audio_to_table(wav_file_path)

    # update config according to currently used model. Should probably happen outside this method
    run_config = config.copy()
    
    run_config['OUTLIER_SIGMA_CUTOFF'] = learn.model_dir.split('_')[-2]
    run_config['REMOVE_OUTLIERS'] = run_config['OUTLIER_SIGMA_CUTOFF'] == 0

    wave_df_proc = run_data_module(run_config, df=wave_df, prod=True)
    row, clas, probs = learn.predict(wave_df_proc.iloc[0])
    
    clas = learn.dls.vocab[clas]
    probs = pd.DataFrame({'class': [learn.dls.vocab[i] for i in range(probs.shape[0])], 'probability': probs}).sort_values('probability', ascending=False)
    
    return wave_df_proc, clas, probs


# + Collapsed="false"
wav_file_path = Path(r'/dsp/dsp_portal/personal/itai.shchorry/AudioClassificationProject/AudioClassifier/Data/UrbanSound8K/audio/fold7/149193-5-0-4.wav')
wave_df_proc, clas, probs = predict_wav(wav_file_path, learn, GLOBAL_CONFIG)

probs.head(3)

# + [markdown] Collapsed="false"
# Let's listen to the wavefile

# + Collapsed="false"
plt.figure(figsize=(12,4))
data,sample_rate = librosa.load(wav_file_path)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(wav_file_path)
