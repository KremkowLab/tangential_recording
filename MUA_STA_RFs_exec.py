#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:29:51 2021

@author: kailun
"""

import numpy as np
import os
import multiprocessing as mp
from MUA_STA_RFs import extract_NP_MUA, get_STA


#%% Parameters

stim_ttl_dir = "/media/kailun/kailun_sata_inte/neuropix-data/2019-07-03_16-11-32/experiment1/recording1/events/Neuropix-PXI-100.0/TTL_1"
raw_data_dir = "/media/kailun/kailun_sata_inte/neuropix-data/2019-07-03_16-11-32/experiment1/recording1/continuous/Neuropix-PXI-100.0"
save_dir = "/home/kailun/Desktop/tangential_recording"
stimulus_path = "/media/kailun/kailun_sata_inte/VisualStimuli/stimuli/locally_light_sparse_noise_36_22_target_size_3_targets_per_frame_2_trials_10_background_0.0_20181120.npy"
stimulus_data = np.load(stimulus_path, allow_pickle=True).item()
sparse_noise_stim = stimulus_data["frames"].astype(float)  # Sparse noise stimuli with shape = (ny, nx, nframes).
STA_lags = np.arange(-7, 3)  # The range of lags for computing STA.
subplots_rc = (24, 16)  # (nrows, ncols) or None. The number of rows and columns for the RF subplots.
fig_fname = "STA_RFs.png"  # The filename for saving the plotted figure.
fig_size_pix = None  # (width, height) or None. The size of the figure (STA RFs) in pixel.

#%% Extract MUA from Neuropixels data

pix_data = extract_NP_MUA(
    stim_ttl_dir,  # The folder path of the stimulus TTLs (path until .../TTL_1).
    raw_data_dir,  # The folder path of raw data (path until .../Neuropix-3a-100.0).
    save_dir,  # The folder path for saving the outputs.
    fname_extensions=(None, None),  # The labels (prefix and postfix) for the files to be saved. Postfix is useful in case there are multiple probes.
    total_ch=384,  # The total number of Neuropixels channels in use.
    spike_event_std_thresh=-4,  # The multiple of standard deviation of Butterworth bandpass filtered signals to be considered as spiking events.
    extract_start_time=None,  # The start time (in unit time) of the data to be extracted. If None, the data will be extracted from the beginning.
    extract_stop_time=None,  # The end time (in unit time) of the data to be extracted. If None, the data will be extracted until the end.
    event_keys=[
        (1, "sync"),
        (2, "starts"),
        (3, "frametimes"),
        (4, "stops"),
    ],  # List containing tuples of channel states (int) and their corresponding TTL keys.
    slice_len=None,  # int, the length (in second) for slicing the data, in case the data size is too big.
    align_to_probe_timestamps=False,  # If True, the stim TTLs in stim_ttl_dir will be aligned to NP probe timestamps (sync TTLs in probe_ttl_dir).
    stim_sync_ch=1,  # The channel state for the sync channel of the stimulus TTLs.
    probe_sync_ch=1,  # The channel state for the sync channel of the probe TTLs.
    probe_ttl_dir="",  # The folder path of the probe TTLs.
    n_cores=int(mp.cpu_count() / 2)  # The number of CPUs to be used for parallel computing. Spare some CPUs so that the computer is not slowed down for other processes.
)

#%% Compute STA and plot the RFs

spiketimes = np.load(os.path.join(save_dir, pix_data.spiketimes_fname), allow_pickle=True).item()
frametimes_dict = np.load(os.path.join(save_dir, pix_data.stim_ttl_fname), allow_pickle=True).item()
LSN_frametimes = frametimes_dict["frametimes"]  # Plese use the key for the stimulus frametimes/TTLs when extracting the NP MUA above.
STA = get_STA(sparse_noise_stim, spiketimes, LSN_frametimes, STA_lags, save_dir)
fig = STA.plot(subplots_rc, fig_fname, fig_size_pix)
