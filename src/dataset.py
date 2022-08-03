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
        self.get_behavior_data(social_network_file = config['DATASET']['BEHAVIOR']['social_network_file_path'],
                               final_file = config['DATASET']['BEHAVIOR']['final_file_path'],
                               midterm_file = config['DATASET']['BEHAVIOR']['midterm_file_path'])
        print(self.final_behavior_data)
        print(self.midterm_behavior_data)
        print(self.social_network_data)
        self.get_eeg_config()
        self.check_data()


    def get_eeg_config(self):
        self.eeg_data = {'final':{'eye_close':{},'eye_open':{}},'midterm':{'eye_close':{},'eye_open':{}}}
        for term in self.eeg_data_path.keys():
            for event in self.eeg_data_path[term].keys():
                for subj_id in self.eeg_data_path[term][event].keys():
                    self.eeg_data[term][event][subj_id] = []
                    for trial in self.eeg_data_path[term][event][subj_id]:
                        raw = mne.io.read_raw_brainvision(trial,preload=self.config['DATASET']['EEG']['preload'])
                        self.eeg_data[term][event][subj_id].append(raw)
    
    def check_data(self):
        self.check = {}
        self.check['subj_id_eeg'] = {}
        for term in ['final','midterm']:
            for event in self.eeg_data_path[term].keys():
                self.check['subj_id_eeg'][f'{term}_{event}'] = list(self.eeg_data_path[term][event].keys())
        eeg_id = list(self.check['subj_id_eeg'])
                            

    def get_behavior_data(self,social_network_file = 'social_network.xlsx',final_file = 'final.xlsx',midterm_file = 'midterm.xlsx'):
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
    
    


                
        
    