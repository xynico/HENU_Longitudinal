from tabnanny import verbose
from typing import final
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from scipy import signal
import mne
from functools import reduce

class SNDataset():

    def __init__(self,config,args):
        self.config = config
        self.args = args
        self.get_eeg_data_path()
        self.get_behavior_data()
        self.get_eeg_config()
        self.check_data()

    def get_eeg_config(self):
        self.eeg_data = {'final':{'eye_close':{},'eye_open':{}},'midterm':{'eye_close':{},'eye_open':{}}}
        for term in self.eeg_data_path.keys():
            for event in self.eeg_data_path[term].keys():
                for subj_id in self.eeg_data_path[term][event].keys():
                    self.eeg_data[term][event][subj_id] = []
                    for trial in self.eeg_data_path[term][event][subj_id]:
                        raw = mne.io.read_raw_brainvision(trial,preload=self.config['DATASET']['EEG']['preload'],verbose = 'ERROR')
                        self.eeg_data[term][event][subj_id].append(raw)

    def check_data(self):
        if self.config['DATASET']['EEG']['check_data'] == 'auto':
            self.check = {}

            # EEG data read
            self.check['subj_id_eeg'] = {}
            for term in ['final','midterm']:
                for event in self.eeg_data_path[term].keys():
                    self.check['subj_id_eeg'][f'{term}_{event}'] = list(self.eeg_data_path[term][event].keys())
            eeg_id = list(self.check['subj_id_eeg'].values())
            self.intersection_eeg_id = reduce(np.intersect1d, eeg_id)
            
            # behavior data read
            self.check['subj_id_behavior'] = {}
            # two terms
            self.check['subj_id_behavior']['final'] = self.final_behavior_data['ID'].to_list()
            self.check['subj_id_behavior']['midterm'] = self.midterm_behavior_data['ID'].to_list()
            behavior_id = list(self.check['subj_id_behavior'].values())
            self.intersection_behavior_id = reduce(np.intersect1d, behavior_id).astype(np.int)

            # social network data read
            id_colums = list(self.social_network_data['ID'])
            id_row = list(self.social_network_data.columns)
            id_row.pop(0)
            self.social_network_id = [id_colums,id_row]

            # check 
            self.intersection_eeg_id = np.array(self.intersection_eeg_id,dtype=str)
            self.intersection_behavior_id = np.array(self.intersection_behavior_id,dtype=str)
            self.used_id = np.intersect1d(self.intersection_eeg_id,self.intersection_behavior_id)
            

        elif type(self.config['DATASET']['EEG']['check_data']) == list:
            self.used_id = self.config['DATASET']['EEG']['check_data']                            
        else:
            raise ValueError('check_data must be either "auto" or list')
        print(f'{len(self.used_id)} subjects are used in the dataset')

    def get_behavior_data(self):

        social_network_file = self.config['DATASET']['BEHAVIOR']['social_network_file_path']
        final_file = self.config['DATASET']['BEHAVIOR']['final_file_path']
        midterm_file = self.config['DATASET']['BEHAVIOR']['midterm_file_path']

        self.final_behavior_data = pd.read_excel(os.path.join(self.args.behavior_data_path,final_file))
        self.midterm_behavior_data = pd.read_excel(os.path.join(self.args.behavior_data_path,midterm_file))
        self.social_network_data = pd.read_excel(os.path.join(self.args.behavior_data_path,social_network_file))

    def get_eeg_data_path(self):
        '''
        self.eeg_data_path = {term:{event:{subject:[repeated_2_trials,],},},}
        '''
        self.eeg_data_path = {'final':{'eye_close':{},'eye_open':{}},'midterm':{'eye_close':{},'eye_open':{}}}
        for term in ['final','midterm']:
            for subj_class in ['even_subj','odd_subj']:
                for event in os.listdir(os.path.join(self.args.eeg_data_path,term,subj_class)):
                    for subj_id in os.listdir(os.path.join(self.args.eeg_data_path,term,subj_class,event)):
                        if not subj_id in self.eeg_data_path[term][event[:-1]]:
                            self.eeg_data_path[term][event[:-1]][subj_id] = [os.path.join(self.args.eeg_data_path,term,subj_class,event,subj_id,subj_id+'.vhdr')]
                        else:
                            self.eeg_data_path[term][event[:-1]][subj_id].append(os.path.join(self.args.eeg_data_path,term,subj_class,event,subj_id,subj_id+'.vhdr'))
        
        # TODO: ordered by event[-1]
    
class EpochDataset():
    
    def __init__(self,raw_dataset,config):
        self.features = {}
        for term in ['final','midterm']:
            for event in raw_dataset[term].keys():
                for subj_id in tqdm(raw_dataset.used_id,desc=f'Loading Epoch dataset {term}_{event}'):
                    for i in range(2):
                        if config['PROCESSING']['Feature_Extraction']['PSD']['DO']:
                            pass




                
        
    