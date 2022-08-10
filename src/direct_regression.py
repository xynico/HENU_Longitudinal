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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from itertools import combinations

class EEG_Regression_trainer():

    def __init__(self,folder_name,dataset,config):
        self.dataset = dataset
        self.config = config
        self.folder_name = folder_name
        if self.config['DO']:
            self.load_EEG_feature()
            self.load_social_network()
            self.direct_regression_EEG()

    def direct_regression_EEG(self):
        check_path(os.path.join(self.config['SAVE_PATH'],self.folder_name))
        self.EEG_model_loop()

    def EEG_model_loop(self):
        self.model = {}
        self.r2_score = {}
        for term in self.dataset.eeg_data.keys():
            self.model[term] = {}
            self.r2_score[term] = {}
            for EEG_method in tqdm(self.EEG_feature[term].keys(),desc=f"{self.config['MODEL']['model_type']}_{term} model "):
                if self.config['PRETRAIN']:
                    self.model[term][EEG_method] = loadpkl(os.path.join(self.config['SAVE_PATH'],self.folder_name,f"{self.config['MODEL']['model_type']}_{term}_{EEG_method}_model.pkl"))
                else:
                    EEG_data = np.stack(self.EEG_feature[term][EEG_method])
                    subj_id_list = self.dataset.used_id
                    X = np.stack([np.concatenate([EEG_data[idx_s1],EEG_data[idx_s2]]) 
                                    for idx_s1 in range(EEG_data.shape[0]) 
                                    for idx_s2 in range(EEG_data.shape[0]) 
                                    if idx_s1 != idx_s2])
                    y = np.stack([self.social_network_feature[np.int(s2)].loc[self.social_network_feature['ID']==np.int(s1)].values[0] for s1 in subj_id_list for s2 in subj_id_list if s1 != s2])
                    X_train,X_val,y_train,y_val = self.split_train_val(X,y)
                    if self.config['MODEL']['standardize']:
                        X_train,X_val = standardize(X_train,X_val)
                    self.model[term][EEG_method] =self.EEG_model_fit(X_train,y_train)
                    savepkl(self.model[term][EEG_method],os.path.join(self.config['SAVE_PATH'],self.folder_name,f"{self.config['MODEL']['model_type']}_{term}_{EEG_method}_model.pkl"))
                self.r2_score[term][EEG_method] = self.EEG_model_test(self.folder_name,X_val,y_val,term,EEG_method)

    def EEG_model_test(self,folder_name,X_val,y_val,term,EEG_method):
        y_pred = self.model[term][EEG_method].predict(X_val)
        r2 = r2_score(y_val,y_pred)
        if self.config['MODEL']['VISUALIZATION']:
            plt.figure(figsize=(10,10))
            plt.scatter(y_val,y_pred,s=1,c='r')
            plt.xlabel('True')
            plt.ylabel('Predicted')
            plt.title(f'True vs Predicted with r^2={r2}')
            plt.savefig(os.path.join(self.config['SAVE_PATH'],self.folder_name,f"{self.config['MODEL']['model_type']}_{term}_{EEG_method}_true_vs_predicted.png"))
            plt.close()
        return r2
                
    def load_EEG_feature(self):
        self.EEG_feature = {term: {} for term in self.dataset.eeg_data.keys()}
        self.EEG_feature_map = {term: {} for term in self.dataset.eeg_data.keys()}
        for term in self.dataset.eeg_data.keys():
            for event in self.dataset.eeg_data[term].keys():
                for method in self.config['EEG_FEATURE']:
                    if method == "PSD":
                        self.EEG_feature[term][f"{event}_{method}"] = []
                        self.EEG_feature_map[term][f"{event}_{method}"] = []
                    elif method == "FUNCTIONALCONNECTIVITY":
                        for fc_method in self.config['EEG_FEATURE']['FUNCTIONALCONNECTIVITY']['method']:
                            self.EEG_feature[term][f"{event}_{method}_{fc_method}"] = []
                            self.EEG_feature_map[term][f"{event}_{method}_{fc_method}"] = []
                for subj_id in self.dataset.used_id:
                    for method in self.config['EEG_FEATURE'].keys():
                        if method == "PSD":
                            psd = loadpkl(os.path.join(self.config['EEG_FEATURE_PATH'],self.folder_name,f'{term}_{event}_{subj_id}_psd.pkl'))
                            psd_vector,psd_vector_loc = flatten_connectivity_matrix(psd[0].mean(0),pop_same=False)# (n_epoch,n_channel,n_psd) -> (n_channel,n_psd) -> (n_psd*n_channel)
                            self.EEG_feature[term][f"{event}_{method}"].append(psd_vector)
                            for idx,(i,j) in enumerate(psd_vector_loc):
                                psd_vector_loc[idx] = (i,psd[1][j])
                            self.EEG_feature_map[term][f"{event}_{method}"].append(psd_vector_loc) #(n_psd*n_channel)
                        elif method == "FUNCTIONALCONNECTIVITY":
                            '''
                            Details could be found in the following link:
                            https://mne.tools/mne-connectivity/stable/auto_examples/mne_inverse_label_connectivity.html#sphx-glr-auto-examples-mne-inverse-label-connectivity-py
                            '''
                            fc = loadpkl(os.path.join(self.config['EEG_FEATURE_PATH'],self.folder_name,f'{term}_{event}_{subj_id}_functional_connectivity.pkl'))
                            con_res = dict()
                            for fc_method, c in zip(self.config['EEG_FEATURE']['FUNCTIONALCONNECTIVITY']['method'], fc):
                                con = c.get_data(output='dense')[:, :, 0]
                                con = np.abs(con)
                                con_vector,con_vector_loc = flatten_connectivity_matrix(con)
                                self.EEG_feature[term][f"{event}_{method}_{fc_method}"].append(con_vector)
                                self.EEG_feature_map[term][f"{event}_{method}_{fc_method}"].append(con_vector_loc)

    

    def load_social_network(self):
        self.social_network_feature = self.dataset.social_network_data
            
    def EEG_model_fit(self,X_train,y_train):
        model=eval(f"{self.config['MODEL']['model_type']}(**{self.config['MODEL']['kwargs']})")
        model.fit(X_train,y_train)
        return model

    def split_train_val(self,X,y):
        X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=self.config['MODEL']['test_rate'],
                                                random_state=self.config['MODEL']['split_random_state'])
        return X_train,X_val,y_train,y_val

    
class Survey_Regression_trainer(EEG_Regression_trainer):

    def __init__(self, folder_name, dataset, config):
        self.dataset = dataset
        self.config = config
        self.folder_name = folder_name
        self.load_social_network()
        self.load_survey_feature()
        self.direct_regression_survey()

    

    def direct_regression_survey(self):
        check_path(os.path.join(self.config['SAVE_PATH'],self.folder_name))
        self.survey_model_loop()
    
    def survey_model_loop(self):
        
        #  GENERATE DATA as a dict: {f"{term}_{event}": [survey_feature,],}
        print("Survey model loop for GENERATE DATA as a dict")
        if self.config['METHOD_WAY']['method'] == "ovo":
            data = {}
            for term in self.survey_feature_map.keys():
                for feature in self.survey_feature_map[term]:
                    data[f"{term}_{feature}"] = np.stack([self.survey_feature[term][feature][int(id)] for id in self.dataset.used_id])
        elif type(self.config['METHOD_WAY']['method']) == dict:
            data = {}
            for term in self.config['METHOD_WAY']['method'].keys():
                for feature in self.config['METHOD_WAY']['method'][term]:
                    data[f"{term}_{feature}"] = np.stack([self.survey_feature[term][feature][int(id)] for id in self.dataset.used_id])
        else:
            raise ValueError("Method way is not defined correctly, which should be either 'ovo' or a dict")
        
        # define the n_feature
        print("Survey model loop for define the n_feature")
        if self.config['METHOD_WAY']['n_feature'] == "all":
            n_feature = range(1,len(data.keys())+1)
        elif type(self.config['METHOD_WAY']['n_feature']) == list:
            n_feature = self.config['METHOD_WAY']['n_feature']
        else:
            raise ValueError("Method way is not defined correctly, which should be either 'all' or a list")

        # Generate the all the possible combination of features
        print("Survey model loop for Generate the all the possible combination of features")
        feature_combination = [list(combinations(data.keys(),n)) for n in n_feature]
        
        # TRAINING LOOP
        self.train(data,feature_combination)
    
    def train(self,data,feature_combination):
        for feature_combination in feature_combination:
            subj_id_list = self.dataset.used_id
            X_feature = 
            X = np.stack([np.concatenate([data[idx_s1],EEG_data[idx_s2]]) 
                            for idx_s1 in range(EEG_data.shape[0]) 
                            for idx_s2 in range(EEG_data.shape[0]) 
                            if idx_s1 != idx_s2])
            y = np.stack([self.social_network_feature[np.int(s2)].loc[self.social_network_feature['ID']==np.int(s1)].values[0] for s1 in subj_id_list for s2 in subj_id_list if s1 != s2])

            X_train,X_val,y_train,y_val = self.split_train_val(data[f"{term}_{feature}"] for term,feature in zip(term,feature))
            model = self.EEG_model_fit(X_train,y_train)
            self.save_model(model,feature_combination)
                
    def load_survey_feature(self):
        self.survey_feature = {"final": self.dataset.final_survey_data,"midterm": self.dataset.midterm_survey_data}
        self.survey_feature_map = {"final": self.dataset.final_survey_data.columns,"midterm": self.dataset.midterm_survey_data.columns}

    
    def survey_model_fit(self,X_train,y_train):
        model=eval(f"{self.config['MODEL']['model_type']}(**{self.config['MODEL']['kwargs']})")
        model.fit(X_train,y_train)
        return model

    