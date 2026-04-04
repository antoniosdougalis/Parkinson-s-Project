# -*- coding: utf-8 -*-
"""
written by Antonios Dougalis, Feb 2026, Kozani Greece
contact: antoniosdougalis (at) gmail.com; antoniosdougalis (at) med.uoc.gr 

"""

# import all required libraries 

import os
import pandas as pd
import sys
import time

import mne
from mne.preprocessing import ICA
 
import numpy as np
import re
import matplotlib.pyplot as plt


def plot_ica_comparison(raw, raw_clean, n_channels=5, tmax=5):
    """
    Plot random channels: raw vs fully cleaned data (ICA + filtering)
    
    raw       : original raw EEG (before ICA)
    raw_clean : cleaned EEG after ICA and filtering
    """
    ch_indices = np.random.choice(len(raw.ch_names), size=n_channels, replace=False)
    times = raw.times[:int(raw.info['sfreq']*tmax)]
    
    raw_data = raw.get_data(picks=ch_indices)[:, :len(times)]
    clean_data = raw_clean.get_data(picks=ch_indices)[:, :len(times)]

    plt.figure(figsize=(12, n_channels*2))
    for i, ch in enumerate(ch_indices):
        plt.subplot(n_channels, 1, i+1)
        plt.plot(times, raw_data[i], label='Raw', alpha=0.6)
        plt.plot(times, clean_data[i], label='Clean (ICA + filter)', linestyle='--')
        plt.ylabel(raw.ch_names[ch])
        if i == 0:
            plt.legend()
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


from scipy.stats import kurtosis
from scipy.signal import welch


def automatic_ica_artifact_muscle(raw, n_components=0.99, method='fastica',
                                  max_iter=1000, power_percentile=95,
                                  kurt_percentile=95, muscle_ratio_thresh=3,
                                  random_state=42):
    """
    ICA with automatic artifact detection including muscle ratio (25-45 Hz / 1-15 Hz).
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG object.
    n_components : float or int
        ICA components (float <1 = PCA variance fraction).
    method : str
        ICA method ('fastica' or 'picard').
    max_iter : int
        Maximum iterations.
    power_percentile : float
        Percentile threshold for component power.
    kurt_percentile : float
        Percentile threshold for component kurtosis.
    muscle_ratio_thresh : float
        Threshold for high/low frequency power ratio.
    random_state : int
        Random seed.
    
    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA object with `ica.exclude` set to artifact ICs.
    """
    
    try:
        ica = ICA(n_components=n_components, method=method,
                  max_iter=max_iter, random_state=random_state, verbose=False)
        ica.fit(raw)
        
    except RuntimeError as e:
        if "One PCA component captures most of the explained variance" in str(e):
            print("Warning: PCA collapsed to 1 component, increasing n_components to full rank.")
            ica = ICA(n_components=None, method=method,
                      max_iter=max_iter, random_state=random_state, verbose=False)
            ica.fit(raw)
        else:
            raise

    # --- get IC time series ---
    sources = ica.get_sources(raw).get_data()  # shape: (n_components, n_times)

    # --- component power ---
    mixing_matrix = ica.mixing_matrix_  # (n_channels, n_components)
    component_power = np.sum(mixing_matrix ** 2, axis=0)

    # --- component kurtosis ---
    component_kurt = kurtosis(sources, axis=1, fisher=False)

    # --- spectral muscle ratio ---
    sfreq = raw.info['sfreq']
    def band_power(sig, fmin, fmax):
        f, pxx = welch(sig, sfreq, nperseg=int(2*sfreq))
        idx = np.logical_and(f >= fmin, f <= fmax)
        return np.trapz(pxx[idx], f[idx])

    muscle_ratio = np.array([
        band_power(sources[c], 25, 45) / band_power(sources[c], 1, 15)
        for c in range(sources.shape[0])
    ])

    # --- thresholds ---
    power_thresh = np.percentile(component_power, power_percentile)
    kurt_thresh = np.percentile(component_kurt, kurt_percentile)

    # --- identify artifact components ---
    artifact_indices = np.where(
        (component_power > power_thresh) |
        (component_kurt > kurt_thresh) |
        (muscle_ratio > muscle_ratio_thresh)
    )[0]

    ica.exclude = artifact_indices.tolist()
    print(f"Identified {len(artifact_indices)} ICA artifact component(s): {ica.exclude}")
    return ica

#%% Start Importations

startTime = time.time()

#% Load raw data for each subject sequentially

#  Initialse the final data containers. This is where the whole data willbe stored
comb_epoched_EEG = None

# Base directory containing the subject folders
base_dir = r'C:\Users\anton\Documents\EEGDATABASE\UCSD_rsEEG_Parkinson\raw_recordings'

# List all directories in base_dir that match the pattern 'sub-hc1-40 or sub-pd1-40'
subject_dirs = [d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and re.match(r'^sub-(hc|pd)([1-9]|[123][0-9]|40)$', d)]

# get the diagnosis lalbesl (0 healthy control (HC), 1 PD)
subj_labels = np.array([1 if 'pd' in subj_id else 0 for subj_id in subject_dirs])


# initialise
file_path = []
epochs_PerSubj = np.zeros( (len(subject_dirs),), dtype =int) 
n_ic_per_subject = np.zeros( (len(subject_dirs),) )  
rejected_ic_per_subject  = []

# Now loop through the found subject folders
for subji, subj_id in enumerate(subject_dirs):
        
    if subji==1:
        break
    
    msg = f'processing subject {subji+1}/of {len(subject_dirs)}, elapsed time {time.time()-startTime:.2f} seconds'
    sys.stdout.write('\r' + msg)
    
    # use only the off medication data for patients
    if 'pd' in subj_id:
        subj_labels[subji] = 1 # store the diagnosis condition label
        
        file_name = f"{subj_id}_ses-on_task-rest_eeg.bdf"
        file_path.append( os.path.join(base_dir, subj_id, 'ses-on','eeg', file_name) )
        
        # read channel information
        chanInfo_name = f"{subj_id}_ses-on_task-rest_channels.tsv"
        temp_chan_info =  pd.read_csv( os.path.join(base_dir, subj_id, 'ses-on','eeg', chanInfo_name) )
        
    
    else:    
        file_name = f"{subj_id}_ses-{subj_id[4:6]}_task-rest_eeg.bdf"
        file_path.append( os.path.join(base_dir, subj_id, 'ses-'+subj_id[4:6],'eeg', file_name) )
        
        # read channel information
        chanInfo_name = f"{subj_id}_ses-{subj_id[4:6]}_task-rest_channels.tsv"
        temp_chan_info =  pd.read_csv( os.path.join(base_dir, subj_id, 'ses-'+subj_id[4:6],'eeg', chanInfo_name) )
        
    
    # load eeg files via mne
    raw = mne.io.read_raw_bdf(file_path[subji], preload=True)
    fs = int(raw.info['sfreq'])
    
    # pick only scalp EEG channels
    scalp_chs = [ch for ch in raw.ch_names if not (ch.startswith('EXG') or ch.startswith('Status'))]
    montage = mne.channels.make_standard_montage('biosemi32')

    raw_eeg = raw.copy().pick_channels(scalp_chs)
    raw_eeg.set_montage(montage)
    
    # set common average reference
    raw_eeg.set_eeg_reference('average', projection=False)
    
    # band-pass filter
    raw_eeg.filter(l_freq=1, h_freq=150, method='fir')
    
    
    # # # OR with additional spectral muscle ratio
    ica = automatic_ica_artifact_muscle(raw_eeg, n_components=0.99, method='fastica',
                                    max_iter=1000, power_percentile=95,
                                    kurt_percentile=95, muscle_ratio_thresh=3)
    
    # reconstruct cleaned signal
    raw_clean = ica.apply(raw_eeg.copy(), verbose=False)
       
    plot_ica_comparison(raw_eeg, raw_clean)
    
    # visualize components for inspection
    # ica.plot_components()
    
    # create 5s epochs with 20% overlap
    epochs = mne.make_fixed_length_epochs(raw_clean, duration=5.0, overlap=1, preload=True)  # 1s = 20% of 5s
    
    # convert to numpy array: epochs x channels x time
    epoched_EEG = epochs.get_data()
    
    # get epochs per Subject
    epochs_PerSubj[subji] = epoched_EEG.shape[0]
    
    # 2: load into containers for saving
    # get the data from each subject and store them
    if comb_epoched_EEG is None:
        comb_epoched_EEG = epoched_EEG
    else:
        comb_epoched_EEG = np.concatenate((comb_epoched_EEG, epoched_EEG), axis=0)
    
    print(f' Data of subj {subj_id} have been processesed and stored: total elapsed time for run is {(time.time()-startTime):.3f} s')

# then create the group labels that would be equal to label equal to total number of epochs
total_epochs = np.sum(epochs_PerSubj)
cum_epochs_PerSubj = np.concatenate( ([0], np.cumsum(epochs_PerSubj)) ).astype(int)

n_subj = len(epochs_PerSubj)
_, n_chans, n_pnts = comb_epoched_EEG.shape

# create the group_labels and the epoch labels variables
group_labels, labels = [ np.zeros((total_epochs,)) for _ in range(2)]

for subji in range(n_subj):
    epS = cum_epochs_PerSubj[subji]
    epE = cum_epochs_PerSubj[subji+1]
    
    tempL = np.ones( (epochs_PerSubj[subji],) )*subj_labels[subji]
    labels[epS:epE] = tempL.copy()
    
    tempG = np.ones( (epochs_PerSubj[subji],) )*subji
    group_labels[epS:epE] = tempG.copy()

 
# Save all data when loop completes
# save_dict = {
    
#     'comb_epoched_EEG': comb_epoched_EEG,
#     'labels': labels,
#     'subj_labels': subj_labels,
#     'group_labels': group_labels,
#     'epochs_PerSubj': epochs_PerSubj,
#     'cum_epochs_PerSubj': cum_epochs_PerSubj,
#     'ch_names':scalp_chs,
#     'fs':fs
    
# }

# os.chdir(r'C:\Users\anton\Documents\Python Scripts\Project_Parkinon_EEG_Analysis')
# np.savez_compressed('sesON_UCSD_Parkinson_EEG_ProcData.npz', **save_dict)
# print('data saved to disk')
