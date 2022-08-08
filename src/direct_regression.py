import mne
import numpy as np
import os
import pandas as pd
from scipy import signal
from tqdm import tqdm
from utils import *


def direct_regression_survey_EEG(folder_name,dataset,config):
    if not os.path.exists(os.path.join(config['SAVE_PATH'],folder_name)): os.makedirs(os.path.join(config['SAVE_PATH'],folder_name))
    EEG_feature = {key:[] for key in config['EEG_FEATURE']}
    for term in dataset.eeg_data.keys():
        for event in dataset.eeg_data[term].keys():
            for subj_id in dataset.eeg_data[term][event].keys():
                

    