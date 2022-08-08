
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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
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
                    savepkl(epoch_concat,os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_{subj_id}_epoch_concat.pkl'))
                
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
                
                
                if config['PSD']['DO']:
                    if not os.path.exists(os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_{subj_id}_psd.pkl')):
                        epoch = loadpkl(os.path.join(config['EPOCH_PATH'],folder_name,f'{term}_{event}_{subj_id}_epoch_concat.pkl'))
                        savepkl(
                        psd(epoch,config['PSD']), 
                        os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_{subj_id}_psd.pkl'))
                if config['FUNCTIONALCONNECTIVITY']['DO']:
                    if not os.path.exists(os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_{subj_id}_functional_connectivity.pkl')):
                        epoch = loadpkl(os.path.join(config['EPOCH_PATH'],folder_name,f'{term}_{event}_{subj_id}_epoch_concat.pkl'))
                        savepkl(
                            functional_connectivity(epoch,config['FUNCTIONALCONNECTIVITY']), 
                            os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_{subj_id}_functional_connectivity.pkl'))

def social_network_loop(folder_name,dataset,config):
    if config['DO']:
        if not os.path.exists(os.path.join(config['SAVE_PATH'],folder_name)): os.makedirs(os.path.join(config['SAVE_PATH'],folder_name))
        for term in dataset.eeg_data.keys():
            for event in dataset.eeg_data[term].keys():
                temp_list = []
                for subj_id in tqdm(dataset.used_id,desc = f"social_network: {term}_{event}"):
                    temp_list.append(social_network(folder_name,config,term,event,subj_id))
                corr_method_dict = {}
                for method in config['FEATURE'].keys():
                    feature_array = np.concatenate([f[method] for f in temp_list],axis = 0)
                    channel_name = [f"{method}_{subj_id}_{i}" for i in range(temp_list[-1][method].shape[0]) for subj_id in dataset.used_id]
                    corr = np.corrcoef(feature_array) # correlation matrix, subj*n_channel x subj*n_channel, indicated by the channel name
                    corr_method_dict[method] = {'corr': corr, 'channel_name': channel_name}
                    if config['VISUALIZATION']:
                        fig,ax = plt.subplots(figsize = (10,10))
                        sns.heatmap(corr,cmap = 'jet',ax = ax)
                        tick_spacing = temp_list[-1][method].shape[0]
                        ax.set_xticks(range(len(channel_name)))
                        ax.set_yticks(range(len(channel_name)))
                        ax.set_xticklabels(channel_name,rotation = 90)
                        ax.set_yticklabels(channel_name)
                        ax.set_title(f"{term}_{event}_{method}")
                        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing+1))
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing+1))
                        plt.savefig(os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_corr.png'))
                        plt.close()
                    
                savepkl(corr_method_dict,os.path.join(config['SAVE_PATH'],folder_name,f'{term}_{event}_corr_method_dict.pkl'))

def social_network(folder_name,config,term,event,subj_id):
    feature = {}
    if 'PSD' in config['FEATURE'].keys():
        psd_array = loadpkl(
            os.path.join(config['FEATURE']['PSD'],folder_name,f'{term}_{event}_{subj_id}_psd.pkl')
            ) # ([epoch,channel,psd],psd)
        feature['PSD'] = np.mean(psd_array[0],axis = 0)
    return feature

    

                    
                    




def psd(epoch,config): 
    
    kwargs = config.copy()
    kwargs.pop('DO')
    return mne.time_frequency.psd_multitaper(epoch,verbose = 'ERROR',**kwargs)

def functional_connectivity(epoch,config):
    kwargs = config.copy()
    kwargs.pop('DO')
    return spectral_connectivity_epochs(epoch,verbose = 'ERROR',**kwargs)

    
