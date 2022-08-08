import mne
import numpy as np
import os
import pandas as pd
from scipy import signal
from tqdm import tqdm
from utils import *


def direct_regression_survey_EEG(folder_name,dataset,config):
    EEG_feature,EEG_feature_map = load_EEG_feature(folder_name,dataset,config)
    survey_feature,survey_feature_map = load_survey_feature(dataset,config)

def matching_features(EEG_feature,EEG_feature_map,survey_feature,survey_feature_map,config):
    '''
    Match the features of EEG and survey data, one vs one
    '''
    feature_matching_dict = {}


def load_EEG_feature(folder_name,dataset,config):
    if not os.path.exists(os.path.join(config['SAVE_PATH'],folder_name)): os.makedirs(os.path.join(config['SAVE_PATH'],folder_name))
    EEG_feature = {}
    EEG_feature_map = {}
    for term in dataset.eeg_data.keys():
        for event in dataset.eeg_data[term].keys():
            for method in config['EEG_FEATURE']:
                if method == "PSD":
                    EEG_feature[f"{term}_{event}_{method}"] = []
                    EEG_feature_map[f"{term}_{event}_{method}"] = []
                elif method == "FUNCTIONALCONNECTIVITY":
                    for fc_method in config['EEG_FEATURE']['FUNCTIONALCONNECTIVITY']['method']:
                        EEG_feature[f"{term}_{event}_{method}_{fc_method}"] = []
                        EEG_feature_map[f"{term}_{event}_{method}_{fc_method}"] = []
            for subj_id in dataset.used_id:
                for method in config['EEG_FEATURE'].keys():
                    if method == "PSD":
                        psd = loadpkl(os.path.join(config['EEG_FEATURE_PATH'],folder_name,f'{term}_{event}_{subj_id}_psd.pkl'))
                        EEG_feature[f"{term}_{event}_{method}"].append(psd[0].mean()) # (n_epoch,n_channel,n_psd) -> (n_channel,n_psd)
                        EEG_feature_map[f"{term}_{event}_{method}"].append(psd[1]) #(n_psd,)
                    elif method == "FUNCTIONALCONNECTIVITY":
                        '''
                        Details could be found in the following link:
                        https://mne.tools/mne-connectivity/stable/auto_examples/mne_inverse_label_connectivity.html#sphx-glr-auto-examples-mne-inverse-label-connectivity-py
                        '''
                        fc = loadpkl(os.path.join(config['EEG_FEATURE_PATH'],folder_name,f'{term}_{event}_{subj_id}_functional_connectivity.pkl'))
                        con_res = dict()
                        for fc_method, c in zip(config['EEG_FEATURE']['FUNCTIONALCONNECTIVITY']['method'], fc):
                            con = c.get_data(output='dense')[:, :, 0]
                            
                            EEG_feature[f"{term}_{event}_{method}_{fc_method}"].append()
                        raise NotImplementedError

                        EEG_feature[f"{term}_{event}_{method}"].append(fc)
    return EEG_feature,EEG_feature_map
def load_survey_feature(dataset,config):
    survey_feature = {"final": dataset.final_survey_data,"midterm": dataset.midterm_survey_data}
    survey_feature_map = {"final": dataset.final_survey_data.columns,"midterm": dataset.midterm_survey_data.columns}
    return survey_feature,survey_feature_map
    

            


                

    