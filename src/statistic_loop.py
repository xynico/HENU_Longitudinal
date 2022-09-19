import numpy as np
import pandas as pd
from utils import *
import pingouin as pg
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
import shap
from search_spaces import *
def statistic_all_in_one(folder_name,trainer,config):
    psd,psd_idx = load_psd(folder_name,trainer,config)
    if config['ASYMMETRY']['DO']: cal_asymmetry_score(folder_name,trainer,config,psd,psd_idx)
    if config['TTEST']['DO']: cal_t_score(folder_name,trainer,config,psd,psd_idx)

def cal_t_score(folder_name,trainer,config,psd,psd_idx):
    # TODO: 
    for term in trainer.dataset.eeg_data.keys():
        for event in trainer.dataset.eeg_data[term].keys():
            psd[f'{term}_{event}'] = psd[f'{term}_{event}']
    


def get_statistical_beta(score,folder_name,trainer,config):
    '''
    score: a dictionary of score, which is dict[f'{term}_{event}'] = (-1,...)
    '''
    if not os.path.exists(os.path.join(config['SAVE_PATH'])):
        os.makedirs(os.path.join(config['SAVE_PATH']))
    subj_id_list = trainer.dataset.used_id
    y = np.stack([trainer.social_network_feature[np.int(idx_s2)].loc[trainer.social_network_feature['ID']==np.int(idx_s1)].values[0] 
                        for idx_s1 in subj_id_list 
                        for idx_s2 in subj_id_list 
                        if not idx_s1 == idx_s2])
    lr = {}
    for term in trainer.dataset.eeg_data.keys():
        for event in trainer.dataset.eeg_data[term].keys():
            X = np.stack([np.concatenate([score[f'{term}_{event}'][idx_s1],score[f'{term}_{event}'][idx_s2]]) 
                                for idx_s1 in range(score[f'{term}_{event}'].shape[0]) 
                                for idx_s2 in range(score[f'{term}_{event}'].shape[0]) 
                                if not idx_s1 == idx_s2])
            # standardize the data
            X = (X - X.mean(0)) / X.std(0)
            lr[f'{term}_{event}'] = pg.linear_regression(X,y,as_dataframe = False)
    lr_df = pd.DataFrame(lr).T
    lr_df.to_csv(os.path.join(config['SAVE_PATH'],f'{folder_name}_lr.csv'))

def get_statistical_beta_all(score,folder_name,trainer,config):
    '''
    score: a dictionary of score, which is dict[f'{term}_{event}'] = (-1,...)
    '''
    subj_id_list = trainer.dataset.used_id
    y = np.stack([trainer.social_network_feature[np.int(idx_s2)].loc[trainer.social_network_feature['ID']==np.int(idx_s1)].values[0] 
                        for idx_s1 in subj_id_list 
                        for idx_s2 in subj_id_list 
                        if not idx_s1 == idx_s2])
    X = []
    X_name = ['Intercept']
    for term in trainer.dataset.eeg_data.keys():
        for event in trainer.dataset.eeg_data[term].keys():
            X.append(np.stack([np.concatenate([score[f'{term}_{event}'][idx_s1],score[f'{term}_{event}'][idx_s2]]) 
                                for idx_s1 in range(score[f'{term}_{event}'].shape[0]) 
                                for idx_s2 in range(score[f'{term}_{event}'].shape[0]) 
                                if not idx_s1 == idx_s2]))
            X_name.append(f'{term}_{event}_A')
            X_name.append(f'{term}_{event}_B')

    X_key = np.stack([f"{idx_s1 < idx_s2}"
                        for idx_s1 in range(score[f'{term}_{event}'].shape[0]) 
                        for idx_s2 in range(score[f'{term}_{event}'].shape[0])
                        if not idx_s1 == idx_s2])

    X = np.concatenate(X,axis=1)
    print(X.shape,X_key.shape)
    # standardize the data
    X = (X - X.mean(0)) / X.std(0)
    lr = pg.linear_regression(X,y)
    lr['names'] = X_name
    lr.to_csv(os.path.join(config['SAVE_PATH'],f'{folder_name}_lr_all.csv'))

    X_train,X_test,y_train,y_test,X_key_train,X_key_val = train_test_split(X,y,X_key,
                                test_size = config['XGBOOST']['test_rate'],
                                random_state=config['XGBOOST']['split_random_state'],
                                stratify = X_key)
    if config['XGBOOST']['BayesianOptimization']['DO']:
        search_spaces = eval(config['XGBOOST']['BayesianOptimization']['search_spaces'])
        model = XGBRegressor(**config['XGBOOST']['kwargs'])
        bayes_cv_tuner = BayesSearchCV(estimator = model,search_spaces = search_spaces,**config['XGBOOST']['BayesianOptimization']['kwargs'])
        bayes_cv_tuner.fit(X_train,y_train)
        model = bayes_cv_tuner.best_estimator_
        set_model = {'model':model,'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test,'X_name':X_name,'bayes_cv_tuner':bayes_cv_tuner}
    else:
        model = XGBRegressor(**config['XGBOOST']['kwargs']).fit(X_train,y_train)
        set_model = {'model':model,'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test,'X_name':X_name}
    savepkl(set_model,os.path.join(config['SAVE_PATH'],f'{folder_name}_xgboost_set_model.pkl'))


def get_xgboost_result(score,folder_name,trainer,config):
    subj_id_list = trainer.dataset.used_id
    y = np.stack([trainer.social_network_feature[np.int(idx_s2)].loc[trainer.social_network_feature['ID']==np.int(idx_s1)].values[0] 
                        for idx_s1 in subj_id_list 
                        for idx_s2 in subj_id_list 
                        if not idx_s1 == idx_s2])
    X = []
    X_name = ['Intercept']
    for term in trainer.dataset.eeg_data.keys():
        for event in trainer.dataset.eeg_data[term].keys():
            X.append(np.stack([np.concatenate([score[f'{term}_{event}'][idx_s1],score[f'{term}_{event}'][idx_s2]]) 
                                for idx_s1 in range(score[f'{term}_{event}'].shape[0]) 
                                for idx_s2 in range(score[f'{term}_{event}'].shape[0]) 
                                if not idx_s1 == idx_s2]))
            X_name.append(f'{term}_{event}_A')
            X_name.append(f'{term}_{event}_B')
    X = np.concatenate(X,axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config['XGBOOST']['split_random_state'])
    # standardize the data
    if config['XGBOOST']['standardize']:
        Ssc = StandardScaler()
        X_train = Ssc.fit_transform(X_train)
        X_test = Ssc.transform(X_test)
    # train the model by Bayesian optimization
    if config['XGBOOST']['bayesian_optimization']:
        pass
        





def cal_asymmetry_score(folder_name,trainer,config,psd,psd_idx):
    asymmetry_score = {}
    for term in trainer.dataset.eeg_data.keys():
        for event in trainer.dataset.eeg_data[term].keys():
            used_psd_idx = np.array(psd_idx >= config['ASYMMETRY']['Band'][0]) * np.array(psd_idx <= config['ASYMMETRY']['Band'][1])
            psd[f'{term}_{event}'] = psd[f'{term}_{event}'][:,:,used_psd_idx].mean(2)
            channel_right = np.isin(config['ASYMMETRY']['CHANNEL_LIST'],config['ASYMMETRY']['CHANNEL_MAP']['right'])
            channel_left = np.isin(config['ASYMMETRY']['CHANNEL_LIST'],config['ASYMMETRY']['CHANNEL_MAP']['left'])
            psd_left = psd[f'{term}_{event}'][:,channel_left].mean(1)
            psd_right = psd[f'{term}_{event}'][:,channel_right].mean(1)
            asymmetry_score[f'{term}_{event}'] = np.log(psd_right).reshape(-1,1) - np.log(psd_left).reshape(-1,1)
    get_statistical_beta(asymmetry_score,folder_name,trainer,config['ASYMMETRY'])
    get_statistical_beta_all(asymmetry_score,folder_name,trainer,config['ASYMMETRY'])

def load_psd(folder_name,trainer,config):
    psd = {}
    for term in trainer.dataset.eeg_data.keys():
        for event in trainer.dataset.eeg_data[term].keys():
            psd_idx = loadpkl(os.path.join(trainer.config['EEG_FEATURE_PATH'],
                    trainer.folder_name,f'{term}_{event}_{trainer.dataset.used_id[0]}_psd.pkl'))[1]
            psd[f'{term}_{event}'] = [loadpkl(os.path.join(trainer.config['EEG_FEATURE_PATH'],
                                                trainer.folder_name,f'{term}_{event}_{subj_id}_psd.pkl'))[0].mean(0)
                                      for subj_id in trainer.dataset.used_id]
            psd[f'{term}_{event}'] = np.stack(psd[f'{term}_{event}'],axis = 0)
    return psd,psd_idx