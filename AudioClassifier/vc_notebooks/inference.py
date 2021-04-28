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

# + Collapsed="false"
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import sys
cwd = Path.cwd()
path_library = str(cwd.parent/'vc_notebooks')
sys.path.append(path_library)

from data import *
from model import *

# + [markdown] Collapsed="false"
# ## Setup

# + Collapsed="false"
# load model

MODEL_FOLDER = cwd.parent/'models'/'tabular'/'tuning_cutoff_0_16'
learn_inf = load_learner(MODEL_FOLDER/'export.pkl')

# load config
GLOBAL_CONFIG = picklel(MODEL_FOLDER/'config.p')

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
wave_df_proc, clas, probs = predict_wav(wav_file_path, learn_inf, GLOBAL_CONFIG)

probs.head(3)

# + [markdown] Collapsed="false"
# ## App

# + [markdown] Collapsed="false"
# Should base our work on [this link](https://github.com/fastai/fastbook/blob/master/02_production.ipynb)

# + Collapsed="false"
# won't work, need to refactor
btn_upload = widgets.FileUpload()
btn_upload

# + Collapsed="false"

