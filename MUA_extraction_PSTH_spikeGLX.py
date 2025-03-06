#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:29:51 2021

@author: kailun
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import MUA_STA_RFs as mua


#%% Parameters

data_dir = '.../dataset'
stim_ttl_dir = os.path.join(data_dir, 'Meta')
raw_data_dir = os.path.join(data_dir, 'MLA-025353_S1_SIM_DCZ_16102024_2_g0/MLA-025353_S1_SIM_DCZ_16102024_2_g0_imec0')
probe_ttl_dir = ""
save_dir = os.path.join(data_dir, 'test_outputs')
raw_data_fname = 'MLA-025353_S1_SIM_DCZ_16102024_2_g0_t0.imec0.ap.bin'


#%% Extract MUA from SpikeGLX data

muaData = mua.neuropixData(
    stim_ttl_dir,  # The folder path of the stimulus TTLs (path until .../TTL_1).
    raw_data_dir,  # The folder path of raw data (path until .../Neuropix-3a-100.0).
    save_dir,  # The folder path for saving the outputs.
    raw_data_fname,   # str or None. The filename of the raw data. If None, 'continuous.dat' will be used.
    fname_extensions=(None, None),  # The labels (prefix and postfix) for the files to be saved. Postfix is useful in case there are multiple probes.
    total_ch=385,  # The total number of channels in use.
    event_keys=[
        (0, "sync"),
        (1, "airpuff"),
    ],  # List containing tuples of channel states (int) and their corresponding TTL keys.
    align_to_probe_timestamps=False,  # If True, the stim TTLs in stim_ttl_dir will be aligned to NP probe timestamps (sync TTLs in probe_ttl_dir).
    stim_sync_ch=0,  # The channel state for the sync channel of the stimulus TTLs.
    probe_sync_ch=1,  # The channel state for the sync channel of the probe TTLs.
    probe_ttl_dir=probe_ttl_dir,  # The folder path of the probe TTLs. Should NOT be the same as stim_ttl_dir.
    isSpikeGlx = True, 
)
muaData.extractMua(
    spike_event_std_thresh=-4,  # The multiple of standard deviation of Butterworth bandpass filtered signals to be considered as spiking events.
    extractStartTimeSec=None,  # The start time (in second, Neuropixels time) of the data to be extracted. If None, the data will be extracted from the beginning.
    extractStopTimeSec=None,  # The end time (in second, Neuropixels time) of the data to be extracted. If None, the data will be extracted until the end.
    sliceLenSec=600,  # int, the length (in second) for slicing the data, in case the data size is too big.
    n_cores=int(os.cpu_count()/2)  # The number of CPUs to be used for parallel computing. Spare some CPUs so that the computer is not slowed down for other processes.
    )


#%% Compute and plot PSTHs

psth_start_sec = -0.1   # the start time of the PSTH in second
psth_end_sec = 0.3   # the stop time of the PSTH in second
psth_interv_sec = 0.01   # the bin size of the PSTH in second
chMapPath = os.path.join(data_dir, 'recording details', 'NP2.0_surface_2_use.imro')

# =============================================================================
# Compute PSTHs
# =============================================================================

st_filename = muaData.spiketimes_fname   # Or "spiketimes.npy" or any other name of the file that contains the spiketimes dictionary
ttl_filename = muaData.stim_ttl_fname   # Or "stim_TTLs.npy" or any other name of the file that contains the stimulus TTLs dictionary
spiketimes = np.load(
    os.path.join(save_dir, st_filename), encoding='latin1', allow_pickle=True).item()
ttlDict = np.load(
    os.path.join(save_dir, ttl_filename), encoding='latin1', allow_pickle=True).item()
ttls = ttlDict["airpuff"]   # Plese use the key for the stimulus frametimes/TTLs when extracting the NP MUA above.
psth_range_sec, psth_range = mua.getPsthRange(
    psth_start_sec, psth_end_sec, psth_interv_sec, muaData._sampling_rate)
psths = mua.getChPsths(spiketimes, ttls, psth_range)

# =============================================================================
# Plot PSTHs
# =============================================================================

plt.subplots()
ax = plt.gca()
mua.plotChPsths(
    ax, psths, psth_range, psth_range_sec, psth_tick_interv=10, xtickRotation=45)
ax.set_title('PSTHS all channels')

chMap = mua.getNpix2ChannelMap(chMapPath)
chMappedPsths = mua.mapPsthsToShank(psths, chMap, emptyVal=np.nan)
chMappedPsthsFig = mua.plotNpix2ChMappedPsths(
    chMappedPsths, psth_range, psth_range_sec, psth_tick_interv=10, 
    sameClim=True, xtickRotation=45, figsize=(10,10))


