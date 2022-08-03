#!/usr/bin/env python
from argparse import ArgumentParser
import sys
from sklearn import datasets
from utils import *
from dataset import *

def main():
    parser = ArgumentParser()
    parser.add_argument('--cfg', type=str, help='config file path')
    parser.add_argument('--eeg_data_path', type=str, default='../data/eed_data', help='eeg data path')
    parser.add_argument('--behavior_data_path', type=str, default='../data/behavior_data', help='behavior data path')
    args = parser.parse_args()
    config = load_config(config_path = args.cfg)
    dataset = SNDataset(config,args)
    

if __name__ == '__main__':
    main()