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

def preprocessing_loop(dataset,config):
    folder_name = f"C{config['CROP']['tmin']}_{config['CROP']['tmax']}_F{config['FILTER']['hp']}_{config['FILTER']['lp']}_N{config['NOTCH_FILTER']['freq']}_E{config['EPOCH']['DurationTime']}"
    if not os.path.exists(os.path.join(config['SAVE_PATH'],folder_name)): os.makedirs(os.path.join(config['SAVE_PATH'],folder_name))
    for term in dataset.eeg_data.keys():
        for event in dataset.eeg_data[term].keys():
            for subj_id in dataset.eeg_data[term][event].keys():
                for i,trial_eeg_raw in enumerate(dataset.eeg_data[term][event][subj_id]):
                    if config['DO']:
                        print(f'{term} {event} {subj_id} {i}')
                        epoch = preprocessing_from_raw(trial_eeg_raw,config)
                        savepkl(epoch,os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_{subj_id}_{i}_epoch.pkl'))
                    else:
                        epoch = loadpkl(os.path.join(config['SAVE_PATH'],f'{term}_{event}_{subj_id}_{i}_epoch.pkl'))





def preprocessing_from_raw(raw,config):

    raw.rename_channels(config['CHANNEL']['mapping'])
    raw.drop_channels(config['CHANNEL']['dropChannel'])
    raw.set_montage(mne.channels.read_custom_montage(config['CHANNEL']['montageName']))
    raw.set_channel_types(config['CHANNEL']['channelTypes'])
    raw.load_data(verbose = 'CRITICAL')
    if config['CROP']['DO']: raw.crop(config['CROP']['tmin'],config['CROP']['tmax'])
    if config['RESAMPLE']['DO']: raw.resample(sfreq = config['RESAMPLE']['sfreq'])
    if config['FILTER']['DO']: raw.filter(config['FILTER']['hp'],config['FILTER']['lp'],n_jobs = config['n_jobs'],verbose = 'CRITICAL')
    if config['NOTCH_FILTER']['DO']: raw.notch_filter(config['NOTCH_FILTER']['freq'],n_jobs = config['n_jobs'],verbose = 'CRITICAL')

    annot = mne.Annotations(onset=list(range(0,int(raw.times[-1]),2)), duration= 0,description=config['EPOCH']['MarkName'])
    raw.set_annotations(annot)
    events,event_key = mne.events_from_annotations(raw)
    epoch = mne.Epochs(raw,preload=True,events = events,event_id=event_key, baseline=(0, 0),tmin = 0.,tmax = config['EPOCH']['DurationTime'])
    if config['ICA']['DO']:
        ransac = Ransac(n_resample = int(raw.info['sfreq']),n_jobs = config['n_jobs'],verbose = False)
        epoch = ransac.fit_transform(epoch)
        ar= AutoReject(random_state=config['ICA']['seed'],thresh_method='random_search',n_jobs = config['n_jobs'],verbose = False)
        epoch = ar.fit_transform(epoch)
        ICAReject = get_rejection_threshold(epoch)
        ica = ICA(random_state = config['ICA']['seed'],method = 'fastica',verbose='CRITICAL')
        ica.fit(epoch,reject=ICAReject)

        # eog_inds,scores = ica.find_bads_eog(epoch) 
        # ica.exclude.extend(eog_inds)
        epoch = ica.apply(epoch)
    if config['REJECT']:
        epoch.pick('eeg')
        reject_criteria = get_rejection_threshold(epoch,decim = 10,cv = 10)
        epoch.drop_bad(reject=reject_criteria)
        epoch.set_eeg_reference(projection=True)
    return epoch




