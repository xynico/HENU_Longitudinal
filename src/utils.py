import pandas as pd
import numpy as np
import os
import yaml
import pickle
from sklearn.preprocessing import StandardScaler

def savepkl(obj,path):
    with open(path,'wb') as f:
        pickle.dump(obj,f)
        
def loadpkl(path):
    with open(path,'rb') as f:
        return pickle.load(f)

def load_config(config_path = 'config.yaml',*kwargs):
    yaml.add_constructor('!join', join)
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    return config

def check_path(path):
    if not os.path.exists(path): 
        os.makedirs(path)
        return 1
    else:
        return 0

def standardize(X_train,X_val):
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        return X_train,X_val

def flatten_connectivity_matrix(con,pop_same = True):
    # flatten the connectivity matrix to 1D and return the location of every element in the matrix as a list
    con_vector = con.flatten()
    con_vector_loc = [(i,j) for i in range(con.shape[0]) for j in range(con.shape[1])]
    if pop_same:
        # remove the diagonal elements
        for idx,(i,j) in enumerate(con_vector_loc):
            if i == j:
                con_vector_loc.remove((i,j))
                con_vector = np.delete(con_vector,idx)
        
    return con_vector,con_vector_loc

## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])
