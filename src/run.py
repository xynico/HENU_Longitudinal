#!/usr/bin/env python
from argparse import ArgumentParser
import sys
from sklearn import datasets
from utils import *
from dataset import *
from EEG_processing import *
from direct_regression import *
from post_feature import *
from statistic_loop import *


def main():
    parser = ArgumentParser()
    parser.add_argument('--cfg', type=str, help='config file path')
    parser.add_argument('--eeg_data_path', type=str, default='../data/eed_data', help='eeg data path')
    parser.add_argument('--survey_data_path', type=str, default='../data/survey_data', help='survey data path')
    args = parser.parse_args()
    config = load_config(config_path = args.cfg)
    dataset = SNDataset(config,args)
    folder_name = preprocessing_loop(dataset,config['PROCESSING']['PREPROCESSING'])
    feature_extraction_loop(folder_name,dataset,config['PROCESSING']['FEATURE_EXTRACTION'])
    EEG_social_network_loop(folder_name,dataset,config['PROCESSING']['EEG_SOCIAL_NETWORK'])
    EEG_dr_trainer = EEG_Regression_trainer(folder_name,dataset,config['PROCESSING']['REGRESSION']['EEG_OVO'])
    survey_dr_trainer = Survey_Regression_trainer(folder_name,dataset,config['PROCESSING']['REGRESSION']['SURVEY_OVO'])
    AIO_trainer = EEG_all_in_one_model_trainer(folder_name,dataset,config['PROCESSING']['REGRESSION']['AIO'])
    post_EEG(folder_name,EEG_dr_trainer,config)
    statistic_all_in_one(folder_name,EEG_dr_trainer,config['STATISTIC'])


        
    

if __name__ == '__main__':
    main()