import mne
import numpy as np
import os
import pandas as pd
from scipy import signal
from tqdm import tqdm
from utils import *

import matplotlib.pyplot as plt
from search_spaces import *


def post_EEG_feature_extraction(folder_name,trainer,config):
    xgboost_feature_importance(folder_name,trainer,config)

def xgboost_feature_importance(folder_name,trainer,config):
    '''
    Extract the feature importance from the model
    '''
    print('Extracting feature importance')
    fscore = {term: {} for term in trainer.dataset.eeg_data.keys()}
    for term in trainer.dataset.eeg_data.keys():
        for model_name in trainer.model[term].keys():
            fscore[term][model_name] = trainer.model[term][model_name].get_booster().get_score(fmap='', importance_type='weight')
            print(f"E {trainer.model[term][model_name].get_booster().get_score(fmap='', importance_type='weight')}"       )
            raise NotImplementedError