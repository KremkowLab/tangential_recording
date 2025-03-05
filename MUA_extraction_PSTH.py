#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 12:09:11 2023

@author: kailun
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import MUA_STA_RFs as mua


#%% Paths

recordingPath = '.../20250224'
stim_ttl_dir = os.path.join(recordingPath, "Openephys2025-02-24_16-21-03/Record Node 103/experiment1/Openephys_Opto_NPX-SC/events/NI-DAQmx-101.PXIe-6341/TTL")
raw_data_dir = os.path.join(recordingPath, "Openephys2025-02-24_16-21-03/Record Node 103/experiment1/Openephys_Opto_NPX-SC/continuous/Neuropix-PXI-100.ProbeA-AP")
probe_ttl_dir = os.path.join(recordingPath, "Openephys2025-02-24_16-21-03/Record Node 103/experiment1/Openephys_Opto_NPX-SC/events/Neuropix-PXI-100.ProbeA-AP/TTL")
save_dir = recordingPath


#%% Extract MUA from Neuropixels data

pix_data = mua.extract_NP_MUA(
    stim_ttl_dir,  # The folder path of the stimulus TTLs (path until .../TTL_1).
    raw_data_dir,  # The folder path of raw data (path until .../Neuropix-3a-100.0).
    save_dir,  # The folder path for saving the outputs.
    fname_extensions=(None, None),  # The labels (prefix and postfix) for the files to be saved. Postfix is useful in case there are multiple probes.
    total_ch=384,  # The total number of Neuropixels channels in use.
    spike_event_std_thresh=-3,  # The multiple of standard deviation of Butterworth bandpass filtered signals to be considered as spiking events.
    extractStartTimeSec=None,  # The start time (in second) of the data to be extracted. If None, the data will be extracted from the beginning.
    extractStopTimeSec=None,  # The end time (in second) of the data to be extracted. If None, the data will be extracted until the end.
    event_keys=[
        (1, "Sync"),
        (4, "Opto"),
        (8, "Camera"),
    ],  # List containing tuples of channel states (int) and their corresponding TTL keys.
    sliceLenSec=600,  # int, the length (in second) for slicing the data, in case the data size is too big.
    align_to_probe_timestamps=True,  # If True, the stim TTLs in stim_ttl_dir will be aligned to NP probe timestamps (sync TTLs in probe_ttl_dir).
    stim_sync_ch=1,  # The channel state for the sync channel of the stimulus TTLs.
    probe_sync_ch=1,  # The channel state for the sync channel of the probe TTLs.
    probe_ttl_dir=probe_ttl_dir,  # The folder path of the probe TTLs.
    n_cores=int(os.cpu_count()/2)  # The number of CPUs to be used for parallel computing. Spare some CPUs so that the computer is not slowed down for other processes.
)


#%% Parameters and data

eventsToPlot = ["Opto"]
psthStartSec = -1
psthEndSec = 2
psthBinSec = 0.01


# =============================================================================
# Load data
# =============================================================================

samplingRate = 30000
stimTtlPath = os.path.join(save_dir, "stim_TTLs.npy")
spiketimePath = os.path.join(save_dir, "spiketimes.npy")
stimTtlDict = np.load(stimTtlPath, allow_pickle=True, encoding='latin1').item()
stDict = np.load(spiketimePath, allow_pickle=True, encoding='latin1').item()
psthRange = np.arange(psthStartSec, psthEndSec, psthBinSec) * samplingRate


# =============================================================================
# Compute PSTHs
# =============================================================================

PSTHs = []
for e, eventKey in enumerate(eventsToPlot):
    ttls = stimTtlDict[eventKey]
    psths = mua.getChPsths(stDict, ttls, psthRange)
    PSTHs.append(psths)


#%% Plotting

nPlot = len(PSTHs)
fig, axs = plt.subplots(1, nPlot, figsize=(5*nPlot, 8))
for i, ax in enumerate(np.array(axs).flatten()):
    plt.sca(ax)
    plt.imshow(PSTHs[i], cmap='magma', origin='lower')
    plt.xticks(np.arange(-0.5, len(psthRange)-0.45, int(round(1/psthBinSec))), 
               np.arange(psthStartSec, psthEndSec+0.1, 1), rotation=45)
    plt.xlabel("Time (s)")
    if i < 1:
        plt.ylabel("Channels")
    else:
        plt.yticks([])
    plt.title(f"{eventsToPlot[i]}")



