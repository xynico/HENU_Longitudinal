import mne
import numpy as np
import os
import pandas as pd
from scipy import signal
from tqdm import tqdm
from utils import *
from sklearn.linear_model import LinearRegression,Ridge
from sklearn import svm
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
def direct_regression_EEG(folder_name,dataset,config):
    if not os.path.exists(os.path.join(config['SAVE_PATH'],folder_name)): os.makedirs(os.path.join(config['SAVE_PATH'],folder_name))
    
    EEG_model_loop(folder_name,dataset,config)

def EEG_model_loop(folder_name,dataset,config):
    EEG_feature,EEG_feature_map = load_EEG_feature(folder_name,dataset,config)
    social_network_feature = load_social_network(dataset,config)
    model_result = {}
    for term in dataset.eeg_data.keys():
        model_result[term] = {}
        for EEG_method in tqdm(EEG_feature[term].keys(),desc=f"{config['MODEL']['model_type']}_{term} model "):
            EEG_data = np.stack(EEG_feature[term][EEG_method])
            subj_id_list = dataset.used_id
            X = np.stack([np.concatenate([EEG_data[idx_s1],EEG_data[idx_s2]]) 
                            for idx_s1 in range(EEG_data.shape[0]) 
                            for idx_s2 in range(EEG_data.shape[0]) 
                            if idx_s1 != idx_s2])
            y = np.stack([social_network_feature[np.int(s2)].loc[social_network_feature['ID']==np.int(s1)].values[0] for s1 in subj_id_list for s2 in subj_id_list if s1 != s2])
            X_train,X_val,y_train,y_val = split_train_val(X,y,config)
            if config['MODEL']['standardize']:
                X_train,X_val = standardize(X_train,X_val)
            model_result[term][EEG_method] =EEG_model_fit(X_train,y_train,config)
            EEG_model_test(folder_name,X_val,y_val,model_result[term][EEG_method],config,term,EEG_method)


def EEG_model_test(folder_name,X_val,y_val,model,config,term,EEG_method):
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val,y_pred)
    if config['MODEL']['VISUALIZATION']:
        plt.figure(figsize=(10,10))
        plt.scatter(y_val,y_pred,s=1,c='r')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(f'True vs Predicted with r^2={r2}')
        plt.savefig(os.path.join(config['SAVE_PATH'],folder_name,f"{config['MODEL']['model_type']}_{term}_{EEG_method}_true_vs_predicted.png"))
        plt.close()
    return model,y_pred


def EEG_model_fit(X_train,y_train,config):
    model=eval(f"{config['MODEL']['model_type']}(**{config['MODEL']['kwargs']})")
    model.fit(X_train,y_train)
    return model

def split_train_val(X,y,config):
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=config['MODEL']['test_rate'],random_state=config['MODEL']['split_random_state'])
    return X_train,X_val,y_train,y_val

def standardize(X_train,X_val):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train,X_val

def load_EEG_feature(folder_name,dataset,config):
    EEG_feature = {term: {} for term in dataset.eeg_data.keys()}
    EEG_feature_map = {term: {} for term in dataset.eeg_data.keys()}
    for term in dataset.eeg_data.keys():
        for event in dataset.eeg_data[term].keys():
            for method in config['EEG_FEATURE']:
                if method == "PSD":
                    EEG_feature[term][f"{event}_{method}"] = []
                    EEG_feature_map[term][f"{event}_{method}"] = []
                elif method == "FUNCTIONALCONNECTIVITY":
                    for fc_method in config['EEG_FEATURE']['FUNCTIONALCONNECTIVITY']['method']:
                        EEG_feature[term][f"{event}_{method}_{fc_method}"] = []
                        EEG_feature_map[term][f"{event}_{method}_{fc_method}"] = []
            for subj_id in dataset.used_id:
                for method in config['EEG_FEATURE'].keys():
                    if method == "PSD":
                        psd = loadpkl(os.path.join(config['EEG_FEATURE_PATH'],folder_name,f'{term}_{event}_{subj_id}_psd.pkl'))
                        psd_vector,psd_vector_loc = flatten_connectivity_matrix(psd[0].mean(0),pop_same=False)# (n_epoch,n_channel,n_psd) -> (n_channel,n_psd) -> (n_psd*n_channel)
                        EEG_feature[term][f"{event}_{method}"].append(psd_vector)
                        for idx,(i,j) in enumerate(psd_vector_loc):
                            psd_vector_loc[idx] = (i,psd[1][j])
                        EEG_feature_map[term][f"{event}_{method}"].append(psd_vector_loc) #(n_psd*n_channel)
                    elif method == "FUNCTIONALCONNECTIVITY":
                        '''
                        Details could be found in the following link:
                        https://mne.tools/mne-connectivity/stable/auto_examples/mne_inverse_label_connectivity.html#sphx-glr-auto-examples-mne-inverse-label-connectivity-py
                        '''
                        fc = loadpkl(os.path.join(config['EEG_FEATURE_PATH'],folder_name,f'{term}_{event}_{subj_id}_functional_connectivity.pkl'))
                        con_res = dict()
                        for fc_method, c in zip(config['EEG_FEATURE']['FUNCTIONALCONNECTIVITY']['method'], fc):
                            con = c.get_data(output='dense')[:, :, 0]
                            con = np.abs(con)
                            con_vector,con_vector_loc = flatten_connectivity_matrix(con)

                            EEG_feature[term][f"{event}_{method}_{fc_method}"].append(con_vector)
                            EEG_feature_map[term][f"{event}_{method}_{fc_method}"].append(con_vector_loc)
    return EEG_feature,EEG_feature_map



def load_survey_feature(dataset,config):
    survey_feature = {"final": dataset.final_survey_data,"midterm": dataset.midterm_survey_data}
    survey_feature_map = {"final": dataset.final_survey_data.columns,"midterm": dataset.midterm_survey_data.columns}
    return survey_feature,survey_feature_map

def load_social_network(dataset,config):
    social_network_feature = dataset.social_network_data
    return social_network_feature
                

    