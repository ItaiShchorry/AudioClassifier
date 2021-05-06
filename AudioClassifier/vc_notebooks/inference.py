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

    
############################### Reference Code - ignore ###############################    
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
# - Model Evaluator
#     - class-level performance:
#         - Allow drill-down to class statistics (PR, F1, num_samples)
#         - Investigate specific populations (FP, FN)
#     - compare 2 models (default - last 2):
#         - compare performance
#         - compare hyperparams
#         
#             
# -


