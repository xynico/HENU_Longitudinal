import mne
import numpy as np
import os
import pandas as pd
from scipy import signal
from tqdm import tqdm
from utils import *

def preprocessing_loop(dataset,config):
    for term in dataset.eeg_data_path.keys():
        for event in dataset.eeg_data_path[term].keys():
            for subj_id in dataset.eeg_data_path[term][event].keys():
                for i,trial_eeg_raw in enumerate(dataset.eeg_data_path[term][event][subj_id]):
                    epoch = preprocessing_from_raw(trial_eeg_raw,config)
                    savepkl(epoch,os.path.join(config['SAVE_PATH'],f'{term}_{event}_{subj_id}_{i}_epoch.pkl'))
                    




def preprocessing_from_raw(raw,config):
    return raw
    




