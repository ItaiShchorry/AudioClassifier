# -*- coding: utf-8 -*-
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
# ## Setup

# + Collapsed="false"
from pathlib import Path
import sys
CWD = Path.cwd()
path_library = str(CWD.parent/'vc_notebooks')
sys.path.append(path_library)

from data import *
from model import *

MODELS_FOLDER = CWD.parent/'models'/'tabular'
MODELS_LST = list(set(get_subdirectories(MODELS_FOLDER)) - set(['.ipynb_checkpoints']))

# + Collapsed="false"
import librosa
def extract_mfcc_features(filepath):
    try:
        audio, sample_rate = librosa.load(filepath, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        st.write("Error encountered while parsing file: ", filepath)
        return None 
     
    return audio, sample_rate, mfccsscaled

def from_audio_to_table(uploaded_wav_file):
    audio, sample_rate, data = extract_mfcc_features(uploaded_wav_file)
    idx = uploaded_wav_file.name

    # Convert into a Panda dataframe 
    X = pd.DataFrame([data], columns=['mfcc_'+str(i) for i in range(data.shape[0])],index=[idx])

    return audio, sample_rate, X

def predict_wav(uploaded_wav_file, learn, config):
    audio, sample_rate, wave_df = from_audio_to_table(uploaded_wav_file)

    # update config according to currently used model. Should probably happen outside this method
    run_config = config.copy()
    
    run_config['OUTLIER_SIGMA_CUTOFF'] = learn.model_dir.split('_')[-2]
    run_config['REMOVE_OUTLIERS'] = run_config['OUTLIER_SIGMA_CUTOFF'] == 0

    wave_df_proc = run_data_module(run_config, df=wave_df, prod=True)
    row, clas, probs = learn.predict(wave_df_proc.iloc[0])
    
    clas = learn.dls.vocab[clas]
    probs = pd.DataFrame({'class': [learn.dls.vocab[i] for i in range(probs.shape[0])], 'probability': probs}).sort_values('probability', ascending=False)
    
    return audio, sample_rate, probs.head()# wave_df_proc, clas, probs

def load_model(path):
    learn_inf = load_learner(path/'export.pkl')
    config = picklel(path/'config.p')
    return learn_inf, config


# + [markdown] Collapsed="false"
# ## Voila Version

# + [markdown] Collapsed="false"
# Should base our work on [this link](https://github.com/fastai/fastbook/blob/master/02_production.ipynb)

# + Collapsed="false"
# # won't work, need to refactor
# btn_upload = widgets.FileUpload()
# btn_upload

# + [markdown] Collapsed="false"
# ## Streamlit Version
#

# + [markdown] Collapsed="false"
# Sections:
# - Model Evaluator
#     - existing metrics
#     - other stuff from error analysis
#     - feature importance distinguishing high/low probabilities in specific class
# - Models Comparer
# - Inference

# + Collapsed="false"
import streamlit as st
def choose_model_helper(text):
    st.write("### Choosing population")  
    model_name = st.selectbox(
        text,
         MODELS_LST)

    model_folder = CWD.parent/'models'/'tabular'/model_name

    learn_inf, config = load_model(model_folder)
    feat_imp = picklel(model_folder/'feat_imp.p')
    
    return learn_inf, config, feat_imp, model_name


# + Collapsed="false"
def get_plotting_dict():
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
    
    return plotting_dict, kinds

# + Collapsed="false" jupyter={"outputs_hidden": true}
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import IPython.display as ipd
st.set_option('deprecation.showPyplotGlobalUse', False)

# home view
st.title("Audio Classification")
st.write("Project for Full Stack Deep Learning course (2021)")

PAGES = ["Data Explorator", "Model Evaluator", "Inferencer"]

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", PAGES)

if selection == "Data Explorator":
    sub_sections = ['Data Summary', 'Target Exploration', 'Correlations']
    internal_selection = st.sidebar.radio("Data Explorator", sub_sections)
    
    learn_inf, config, _, _ = choose_model_helper('Which dataset would you like to examine?')
    df = read_df_from_path(config['FILE_PATH'])
    
    if internal_selection == 'Data Summary':
        st.write("### Sample data")
        st.write(df.head())
        
        st.write("### Column-level summary")
        st.write("We'll summarize some meta-level information + aggregative measures for each column")
        st.write(create_col_level_summary_table(df))
        st.write("This summary can already give us some insights on the data and actions that we'd like to take, e.g. drill-down a high-cardinality column or drop one with many missing values. Those will be part of the next section, which handles data validation & data transformations as a preparation for running the model.")
    
    if internal_selection == 'Target Exploration':
        fig, ax = plt.subplots(figsize=(14,6))
        sns.countplot(x=config['TARGET'], data=df,ax=ax)
        st.pyplot()
    
    
        st.write("### Relationship with target")
        st.write("A quick way to visualize interactions with the target")
        plotting_dict, kinds = get_plotting_dict()

        feat = st.selectbox("Feature", list(df.columns))
        plot = st.selectbox("Plot", kinds)

        plotting_dict[plot]['f'](x=feat, y=config['TARGET'], data=df, **plotting_dict[plot]['kwargs'])
        st.pyplot()

    if internal_selection == 'Correlations':
        st.write("### Correlation Heatmap")
        st.write("Can be divided into:")
        st.write("- **Linear** correlation between the variables - Pearson")
        st.write("- **Monotonous** correlation between the variables (AKA Ranked correlation) - Spearman")
        st.write("- **Ordinal association** between the variables (variables are growing/decreasing together) - Kendall")

        corr_method = st.selectbox("Method", ['spearman','pearson','kendall'])
        corr_table = df.corr(method=corr_method,min_periods=int(df.shape[0]*0.5))
        heatmap = sns.heatmap(corr_table)
        st.pyplot()
        
        st.write("### Hierarchical Clustering")
        flat_corr_table = scipy.cluster.hierarchy.distance.squareform((1-corr_table))
        hac = linkage(flat_corr_table, optimal_ordering=True)

        fig = plt.figure(figsize=(10,6))
        dendrogram(hac, labels=corr_table.columns)
        st.pyplot()
        
if selection == "Model Evaluator":

    # create performance df

    perf_dfs = []
    for m in MODELS_LST:
        m_path = CWD.parent/'models'/'tabular'
        perf_dfs.append(picklel(m_path/m/'metrics_df.p'))

    perf_dfs = pd.concat(perf_dfs)

    st.write("## General Performance")  
    glob_stats = ['tr_acc','val_acc','test_acc']
    st.write(perf_dfs[['tr_acc','val_acc','test_acc']])
    
    st.write("## Class-level Performance")
    cs = list(set(perf_dfs.columns) - set(glob_stats))
    sns.relplot(data=perf_dfs[cs], kind='line')
    st.pyplot()

    st.write("## Model Drilldown") 
    # choose model to evaluate
    learn_inf, config, feat_imp, _ = choose_model_helper('Which model would you like to investigate?')
    st.write("### Feature Importance") 
    st.write("implemented via permutation importance methodology") 
    st.write(feat_imp)

if selection == "Inferencer":
    learn_inf, config, _, model_name = choose_model_helper('Which model would you like to predict with?')
    
    st.write("## Inference")
    uploaded_wav = st.file_uploader('upload a wave file for prediction')

    if uploaded_wav is not None:
        
        audio, sample_rate, probs = predict_wav(uploaded_wav, learn_inf, config)
        
        
        st.write("You uploaded this sound")
        html_object = ipd.Audio(audio, rate=sample_rate)
        print(type(html_object))
        st.components.v1.html(html_object._repr_html_())
        
        st.write(f"And the classification probabilities using {model_name} are:")
        st.write(probs)

    
############################### Reference Code ###############################    
# creating a slider
# x = st.slider('Slope', min_value=0.01, max_value=0.10, step=0.01)
# y = st.slider('Noise', min_value=0.01, max_value=0.10, step=0.01)

# can also create these in a sidebar
# st.sidebar.markdown("## Controls")
# st.sidebar.markdown("you can **change** the values to change the charts")
# x = st.sidebar.slider('Slope', min_value=0.01, max_value=0.10, step=0.01)
# y = st.sidebar.slider('Noise', min_value=0.01, max_value=0.10, step=0.01)

# @st.cache
# def create_values(x,y):
#     return np.cumprod(1+np.random.normal(x,y,(100,10)),axis=0)

# # creating a line chart
# if st.sidebar.checkbox("Toggle plot", True):
#     values = create_values(x,y)
#     st.line_chart(values)

# show a piece of code to the user
# with st.echo():
#     print("show this code")



# creating charts with matplotlib
# fig, ax = plt.subplots()
# plt.title(f"x={x}, y={y}")
# for i in range(values.shape[1]):
#     plt.plot(values[:, i])

# st.pyplot(fig)

# + [markdown] Collapsed="false"
# ### Backlog
# - Enable training
#     - interactive preprocessing:
#         - expose preprocessing configuration
#         - outlier detection thresholder
#         - 

# + [markdown] Collapsed="false"
# ### MadeWithML

# + Collapsed="false"
# # streamlit/st_app.py
# # Streamlit application.

# import itertools
# from collections import Counter, OrderedDict
# from distutils.util import strtobool
# from pathlib import Path

# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns

# import streamlit as st
# from app import cli


# @st.cache
# def get_performance(model_dir):
#     """
#     How would we display performance?
#     Global + per-class?
#     If so, should save results for each deployed model
#     """
#     performance = utils.load_dict(filepath=Path(model_dir, "performance.json"))
#     return performance


# @st.cache
# def get_tags(author=config.AUTHOR, repo=config.REPO):
#     # Get list of tags
#     tags_list = ["workspace"] + [
#         tag["name"]
#         for tag in utils.load_json_from_url(
#             url=f"https://api.github.com/repos/{author}/{repo}/tags"
#         )
#     ]

#     # Get metadata by tag
#     tags = {}
#     for tag in tags_list:
#         tags[tag] = {}
#         tags[tag]["params"] = cli.params(tag=tag, verbose=False)
#         tags[tag]["performance"] = pd.json_normalize(
#             cli.performance(tag=tag, verbose=False), sep="."
#         ).to_dict(orient="records")[0]

#     return tags


# @st.cache
# def get_diff(author=config.AUTHOR, repo=config.REPO, tag_a="workspace", tag_b=""):
#     params_diff, performance_diff = cli.diff(author=author, repo=repo, tag_a=tag_a, tag_b=tag_b)
#     return params_diff, performance_diff


# @st.cache
# def get_artifacts(run_id):
#     artifacts = main.load_artifacts(run_id=run_id)
#     return artifacts


# @st.cache
# def evaluate_df(df, tags_dict, artifacts):
#     # Retrieve artifacts
#     params = artifacts["params"]

#     # Prepare
#     df, tags_above_freq, tags_below_freq = data.prepare(
#         df=df,
#         include=list(tags_dict.keys()),
#         exclude=config.EXCLUDED_TAGS,
#         min_tag_freq=int(params.min_tag_freq),
#     )

#     # Preprocess
#     df.text = df.text.apply(
#         data.preprocess,
#         lower=bool(strtobool(str(params.lower))),
#         stem=bool(strtobool(str(params.stem))),
#     )

#     # Evaluate
#     y_true, y_pred, performance = eval.evaluate(df=df, artifacts=artifacts)
#     sorted_tags = list(
#         OrderedDict(
#             sorted(performance["class"].items(), key=lambda tag: tag[1]["f1"], reverse=True)
#         ).keys()
#     )

#     return y_true, y_pred, performance, sorted_tags, df


# # Title
# st.title("Tagifai · MLOps · Made With ML")
# """by [Goku Mohandas](https://twitter.com/GokuMohandas)"""
# st.info("🔍 Explore the different pages below.")

# # Pages
# pages = ["Data", "Performance", "Inference", "Inspection"]
# st.header("Pages")
# selected_page = st.radio("Select a page:", pages, index=2)

# if selected_page == "Data":
#     st.header("Data")

#     # Load data
#     projects_fp = Path(config.DATA_DIR, "projects.json")
#     tags_fp = Path(config.DATA_DIR, "tags.json")
#     projects = utils.load_dict(filepath=projects_fp)
#     tags_dict = utils.list_to_dict(utils.load_dict(filepath=tags_fp), key="tag")
#     col1, col2 = st.beta_columns(2)
#     with col1:
#         st.subheader("Projects (sample)")
#         st.write(projects[0])
#     with col2:
#         st.subheader("Tags")
#         tag = st.selectbox("Choose a tag", list(tags_dict.keys()))
#         st.write(tags_dict[tag])

#     # Dataframe
#     df = pd.DataFrame(projects)
#     st.text(f"Projects (count: {len(df)}):")
#     st.write(df)

#     # Filter tags
#     st.write("---")
#     st.subheader("Annotation")
#     st.write(
#         "We want to determine what the minimum tag frequency is so that we have enough samples per tag for training."
#     )
#     min_tag_freq = st.slider("min_tag_freq", min_value=1, value=30, step=1)
#     df, tags_above_freq, tags_below_freq = data.prepare(
#         df=df,
#         include=list(tags_dict.keys()),
#         exclude=config.EXCLUDED_TAGS,
#         min_tag_freq=min_tag_freq,
#     )
#     col1, col2, col3 = st.beta_columns(3)
#     with col1:
#         st.write("**Most common tags**:")
#         for item in tags_above_freq.most_common(5):
#             st.write(item)
#     with col2:
#         st.write("**Tags that just made the cut**:")
#         for item in tags_above_freq.most_common()[-5:]:
#             st.write(item)
#     with col3:
#         st.write("**Tags that just missed the cut**:")
#         for item in tags_below_freq.most_common(5):
#             st.write(item)
#     with st.beta_expander("Excluded tags"):
#         st.write(config.EXCLUDED_TAGS)

#     # Number of tags per project
#     st.write("---")
#     st.subheader("Exploratory Data Analysis")
#     num_tags_per_project = [len(tags) for tags in df.tags]
#     num_tags, num_projects = zip(*Counter(num_tags_per_project).items())
#     plt.figure(figsize=(10, 3))
#     ax = sns.barplot(list(num_tags), list(num_projects))
#     plt.title("Tags per project", fontsize=20)
#     plt.xlabel("Number of tags", fontsize=16)
#     ax.set_xticklabels(range(1, len(num_tags) + 1), rotation=0, fontsize=16)
#     plt.ylabel("Number of projects", fontsize=16)
#     plt.show()
#     st.pyplot(plt)

#     # Distribution of tags
#     tags = list(itertools.chain.from_iterable(df.tags.values))
#     tags, tag_counts = zip(*Counter(tags).most_common())
#     plt.figure(figsize=(10, 3))
#     ax = sns.barplot(list(tags), list(tag_counts))
#     plt.title("Tag distribution", fontsize=20)
#     plt.xlabel("Tag", fontsize=16)
#     ax.set_xticklabels(tags, rotation=90, fontsize=14)
#     plt.ylabel("Number of projects", fontsize=16)
#     plt.show()
#     st.pyplot(plt)

#     # Preprocessing
#     st.write("---")
#     st.subheader("Preprocessing")
#     text = st.text_input("Input text", "Conditional generation using Variational Autoencoders.")
#     filters = st.text_input("filters", "[!\"'#$%&()*+,-./:;<=>?@\\[]^_`{|}~]")
#     lower = st.checkbox("lower", True)
#     stem = st.checkbox("stem", False)
#     preprocessed_text = data.preprocess(text=text, lower=lower, stem=stem, filters=filters)
#     st.text("Preprocessed text:")
#     st.write(preprocessed_text)

# elif selected_page == "Performance":
#     st.header("Performance")

#     # Get tags and respective parameters and performance
#     tags = get_tags(author=config.AUTHOR, repo=config.REPO)

#     # Key metrics
#     key_metrics = [
#         "overall.f1",
#         "overall.precision",
#         "overall.recall",
#         "behavioral.score",
#         "slices.overall.f1",
#         "slices.overall.precision",
#         "slices.overall.recall",
#     ]

#     # Key metric values over time
#     key_metrics_over_time = {}
#     for metric in key_metrics:
#         key_metrics_over_time[metric] = {}
#         for tag in tags:
#             key_metrics_over_time[metric][tag] = tags[tag]["performance"][metric]
#     st.line_chart(key_metrics_over_time)

#     # Compare two performance
#     st.subheader("Compare performances:")
#     d = {}
#     col1, col2 = st.beta_columns(2)
#     with col1:
#         tag_a = st.selectbox("Tag A", list(tags.keys()), index=0)
#         d[tag_a] = {"links": {}}
#     with col2:
#         tag_b = st.selectbox("Tag B", list(tags.keys()), index=1)
#         d[tag_b] = {"links": {}}
#     if tag_a == tag_b:
#         raise Exception("Tags must be different in order to compare them.")

#     # Diffs
#     params_diff, performance_diff = get_diff(
#         author=config.AUTHOR, repo=config.REPO, tag_a=tag_a, tag_b=tag_b
#     )
#     with st.beta_expander("Key metrics", expanded=True):
#         key_metrics_dict = {metric: performance_diff[metric] for metric in key_metrics}
#         key_metrics_diffs = [key_metrics_dict[metric]["diff"] * 100 for metric in key_metrics_dict]
#         plt.figure(figsize=(10, 5))
#         ax = sns.barplot(
#             x=key_metrics,
#             y=key_metrics_diffs,
#             palette=["green" if value >= 0 else "red" for value in key_metrics_diffs],
#         )
#         ax.axhline(0, ls="--")
#         for i, (metric, value) in enumerate(zip(key_metrics, key_metrics_diffs)):
#             ax.annotate(
#                 s=f"{value:.2f}%\n({key_metrics_dict[metric][tag_a]:.2f} / {key_metrics_dict[metric][tag_b]:.2f})\n\n",
#                 xy=(i, value),
#                 ha="center",
#                 va="center",
#                 xytext=(0, 10),
#                 textcoords="offset points",
#                 fontsize=12,
#             )
#         sns.despine(ax=ax, bottom=False, left=False)
#         plt.xlabel("Metric", fontsize=16)
#         ax.set_xticklabels(key_metrics, rotation=30, fontsize=10)
#         plt.ylabel("Diff (%)", fontsize=16)
#         plt.show()
#         st.pyplot(plt)

#     with st.beta_expander("Hyperparameters"):
#         st.json(params_diff)
#     with st.beta_expander("Improvements"):
#         st.json({metric: value for metric, value in performance_diff.items() if value["diff"] >= 0})
#     with st.beta_expander("Regressions"):
#         st.json({metric: value for metric, value in performance_diff.items() if value["diff"] < 0})

# elif selected_page == "Inference":
#     st.header("Inference")
#     text = st.text_input(
#         "Enter text:",
#         "Transfer learning with transformers for self-supervised learning.",
#     )
#     prediction = cli.predict_tags(text=text)
#     st.text("Prediction:")
#     st.write(prediction)

# elif selected_page == "Inspection":
#     st.header("Inspection")
#     st.write(
#         "We're going to inspect the TP, FP and FN samples across our different tags. It's a great way to catch issues with labeling (FP), weaknesses (FN), etc."
#     )

#     # Load data
#     projects_fp = Path(config.DATA_DIR, "projects.json")
#     tags_fp = Path(config.DATA_DIR, "tags.json")
#     projects = utils.load_dict(filepath=projects_fp)
#     tags_dict = utils.list_to_dict(utils.load_dict(filepath=tags_fp), key="tag")
#     df = pd.DataFrame(projects)

#     # Get performance
#     run_id = open(Path(config.MODEL_DIR, "run_id.txt")).read()
#     artifacts = get_artifacts(run_id=run_id)
#     label_encoder = artifacts["label_encoder"]
#     y_true, y_pred, performance, sorted_tags, df = evaluate_df(
#         df=df,
#         tags_dict=tags_dict,
#         artifacts=artifacts,
#     )
#     tag = st.selectbox("Choose a tag", sorted_tags, index=0)
#     st.json(performance["class"][tag])

#     # TP, FP, FN samples
#     index = label_encoder.class_to_index[tag]
#     tp, fp, fn = [], [], []
#     for i in range(len(y_true)):
#         true = y_true[i][index]
#         pred = y_pred[i][index]
#         if true and pred:
#             tp.append(i)
#         elif not true and pred:
#             fp.append(i)
#         elif true and not pred:
#             fn.append(i)

#     # Samples
#     num_samples = 3
#     with st.beta_expander("True positives"):
#         if len(tp):
#             for i in tp[:num_samples]:
#                 st.write(f"{df.text.iloc[i]}")
#                 st.text("True")
#                 st.write(label_encoder.decode([y_true[i]])[0])
#                 st.text("Predicted")
#                 st.write(label_encoder.decode([y_pred[i]])[0])
#     with st.beta_expander("False positives"):
#         if len(fp):
#             for i in fp[:num_samples]:
#                 st.write(f"{df.text.iloc[i]}")
#                 st.text("True")
#                 st.write(label_encoder.decode([y_true[i]])[0])
#                 st.text("Predicted")
#                 st.write(label_encoder.decode([y_pred[i]])[0])
#     with st.beta_expander("False negatives"):
#         if len(fn):
#             for i in fn[:num_samples]:
#                 st.write(f"{df.text.iloc[i]}")
#                 st.text("True")
#                 st.write(label_encoder.decode([y_true[i]])[0])
#                 st.text("Predicted")
#                 st.write(label_encoder.decode([y_pred[i]])[0])
#     st.write("\n")
#     st.warning(
#         "Be careful not to make decisions based on predicted probabilities before [calibrating](https://arxiv.org/abs/1706.04599) them to reliably use as measures of confidence."
#     )
#     """
#     ### Extensions

#     - Use false positives to identify potentially mislabeled data.
#     - Connect inspection pipelines with annotation systems so that changes to the data can be reviewed and incorporated.
#     - Inspect FP / FN samples by [estimating training data influences (TracIn)](https://arxiv.org/abs/2002.08484) on their predictions.
#     - Inspect the trained model's behavior under various conditions using the [WhatIf](https://pair-code.github.io/what-if-tool/) tool.
#     """


# else:
#     st.text("Please select a valid page option from above...")

# st.write("---")

# # Resources
# """
# ## Resources

# - 🎓 Lessons: https://madewithml.com/
# - 🐙 Repository: https://github.com/GokuMohandas/MLOps
# - 📘 Documentation: https://gokumohandas.github.io/mlops/
# - 📬 Subscribe: https://newsletter.madewithml.com
# """
