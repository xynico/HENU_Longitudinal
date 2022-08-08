import mne
import numpy as np
import os
import pandas as pd
from scipy import signal
from tqdm import tqdm
from utils import *


def direct_regression_survey_EEG(folder_name,dataset,config):
    EEG_feature,EEG_feature_map = load_EEG_feature(folder_name,dataset,config)


def load_EEG_feature(folder_name,dataset,config):
    if not os.path.exists(os.path.join(config['SAVE_PATH'],folder_name)): os.makedirs(os.path.join(config['SAVE_PATH'],folder_name))
    EEG_feature = {term: {} for term in dataset.eeg_data.keys()}
    EEG_feature_map = {term: {} for term in dataset.eeg_data.keys()}
    for term in dataset.eeg_data.keys():
        for event in dataset.eeg_data[term].keys():
            EEG_feature[term][event] = {key:[] for key in config['EEG_FEATURE']}
            EEG_feature_map[term][event] = {key:[] for key in config['EEG_FEATURE']}
            for subj_id in dataset.used_id:
                for method in config['EEG_FEATURE']:
                    if method == "PSD":
                        psd = loadpkl(os.path.join(config['EEG_FEATURE_PATH'],folder_name,f'{term}_{event}_{subj_id}_psd.pkl'))
                        EEG_feature[term][event][method].append(psd[0].mean()) # (n_epoch,n_channel,n_psd) -> (n_channel,n_psd)
                        EEG_feature_map[term][event][method].append(psd[1]) #(n_psd,)
                    elif method == "FUNCTIONALCONNECTIVITY":
                        raise ValueError("TODO: functional connectivity")
    return EEG_feature,EEG_feature_map
def load_survey_feature(dataset,config):
    survey_feature = dataset.final_behavior_data
    

            


                

    