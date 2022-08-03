import pandas as pd
import numpy as np
import os
import yaml
import pickle

def savepkl(obj,path):
    with open(path,'wb') as f:
        pickle.dump(obj,f)
        
def loadpkl(path):
    with open(path,'rb') as f:
        return pickle.load(f)

def load_config(config_path = 'config.yaml',*kwargs):
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    return config


##### AB




