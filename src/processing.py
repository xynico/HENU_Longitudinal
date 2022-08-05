from errno import EREMCHG
from tabnanny import verbose
import mne
import numpy as np
import os
import pandas as pd
from scipy import signal
from tqdm import tqdm
from utils import *
from autoreject.utils import interpolate_bads
from autoreject import AutoReject,set_matplotlib_defaults
from autoreject import Ransac,get_rejection_threshold
from mne.preprocessing import ICA
from mne_connectivity import spectral_connectivity_epochs

def get_folder_name(config):
    folder_name = f"C{config['CROP']['tmin']}_{config['CROP']['tmax']}_F{config['FILTER']['hp']}_{config['FILTER']['lp']}_N{config['NOTCH_FILTER']['freq']}_E{config['EPOCH']['DurationTime']}"
    return folder_name

def preprocessing_loop(dataset,config):
    folder_name = get_folder_name(config)
    if not os.path.exists(os.path.join(config['SAVE_PATH'],folder_name)): os.makedirs(os.path.join(config['SAVE_PATH'],folder_name))
    for term in dataset.eeg_data.keys():
        for event in dataset.eeg_data[term].keys():
            for subj_id in tqdm(dataset.used_id,desc = f"Preprocessing: {term}_{event}"):
                if not os.path.exists(os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_{subj_id}_epoch_concat.pkl')):
                    epoch_list = []
                    for i,trial_eeg_raw in enumerate(dataset.eeg_data[term][event][subj_id]):
                        if config['DO']:
                            epoch = preprocessing_from_raw(trial_eeg_raw,config)
                            savepkl(epoch,os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_{subj_id}_{i}_epoch.pkl'))
                        else:
                            if not os.path.exists(os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_{subj_id}_{i}_epoch.pkl')):
                                epoch = preprocessing_from_raw(trial_eeg_raw,config)
                                savepkl(epoch,os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_{subj_id}_{i}_epoch.pkl'))
                            else:
                                    epoch = loadpkl(os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_{subj_id}_{i}_epoch.pkl'))
                        epoch_list.append(epoch)
                        epoch_concat = mne.concatenate_epochs(epoch_list,verbose = 'ERROR')
                        savepkl(epoch_list,os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_{subj_id}_epoch_concat.pkl'))
                
    return folder_name

def preprocessing_from_raw(raw,config):

    raw.rename_channels(config['CHANNEL']['mapping'],verbose = 'ERROR')
    raw.drop_channels(config['CHANNEL']['dropChannel'])
    raw.set_montage(mne.channels.read_custom_montage(config['CHANNEL']['montageName']),verbose = 'ERROR')
    raw.set_channel_types(config['CHANNEL']['channelTypes'],verbose = 'ERROR')
    raw.load_data(verbose = 'ERROR')
    try:
        if config['CROP']['DO']: raw.crop(config['CROP']['tmin'],config['CROP']['tmax'],verbose = 'ERROR')
    except:
        if config['CROP']['DO']: raw.crop(config['CROP']['tmin'],verbose = 'ERROR')
    if config['RESAMPLE']['DO']: raw.resample(sfreq = config['RESAMPLE']['sfreq'],verbose = 'ERROR')
    if config['FILTER']['DO']: raw.filter(config['FILTER']['hp'],config['FILTER']['lp'],n_jobs = config['n_jobs'],verbose = 'ERROR')
    if config['NOTCH_FILTER']['DO']: raw.notch_filter(config['NOTCH_FILTER']['freq'],n_jobs = config['n_jobs'],verbose = 'ERROR')

    annot = mne.Annotations(onset=list(range(0,int(raw.times[-1]),2)), duration= 0,description=config['EPOCH']['MarkName'])
    raw.set_annotations(annot,verbose = 'ERROR')
    events,event_key = mne.events_from_annotations(raw,verbose = 'ERROR')
    epoch = mne.Epochs(raw,preload=True,events = events,event_id=event_key, baseline=(0, 0),tmin = 0.,tmax = config['EPOCH']['DurationTime'],verbose = 'ERROR')
    if config['ICA']['DO']:
        ransac = Ransac(n_resample = int(raw.info['sfreq']),n_jobs = config['n_jobs'],verbose = False)
        epoch = ransac.fit_transform(epoch)
        ar= AutoReject(random_state=config['ICA']['seed'],thresh_method='random_search',n_jobs = config['n_jobs'],verbose = False)
        epoch = ar.fit_transform(epoch)
        ICAReject = get_rejection_threshold(epoch,verbose = False)
        ica = ICA(random_state = config['ICA']['seed'],method = 'fastica',verbose = 'ERROR')
        ica.fit(epoch,reject=ICAReject,verbose = 'ERROR')

        # eog_inds,scores = ica.find_bads_eog(epoch) 
        # ica.exclude.extend(eog_inds)

        epoch = ica.apply(epoch)
    
    if config['REJECT']:
        epoch.pick('eeg')
        reject_criteria = get_rejection_threshold(epoch,decim = 10,cv = 10)
        epoch.drop_bad(reject=reject_criteria)
        epoch.set_eeg_reference(projection=True)
    
    return epoch

def feature_extraction_loop(folder_name,dataset,config):
    
    if not os.path.exists(os.path.join(config['SAVE_PATH'],folder_name)): os.makedirs(os.path.join(config['SAVE_PATH'],folder_name))
    for term in dataset.eeg_data.keys():
        for event in dataset.eeg_data[term].keys():
            for subj_id in  tqdm(dataset.used_id,desc = f"feature extraction: {term}_{event}"):
                epoch = loadpkl(os.path.join(config['EPOCH_PATH'],folder_name,f'{term}_{event}_{subj_id}_epoch_concat.pkl',))
                if config['PSD']['DO']:
                    savepkl(
                        psd(epoch,config), 
                        os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_{subj_id}_{i}_psd.pkl'))
                if config['FUNCTIONALCONNECTIVITY']['DO']:
                    savepkl(
                        functional_connectivity(epoch,config), 
                        os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_{subj_id}_{i}_functional_connectivity.pkl'))
                    

def psd(epoch,config): 
    return mne.time_frequency.psd_multitaper(epoch,verbose = 'ERROR',**config['PSD'])

def functional_connectivity(epoch,config):
    return spectral_connectivity_epochs(epoch,verbose = 'ERROR',**config['FUNCTIONALCONNECTIVITY'])


