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


#%% Parameters and paths

mapRf = True
computePsth = False
recordingPath = '.../example_data'
stim_ttl_dir = os.path.join(recordingPath, "2019-07-18_13-16-40/2019-07-18_13-16-40/experiment1/recording1/events/Neuropix-PXI-100.0/TTL_1")
raw_data_dir = os.path.join(recordingPath, "2019-07-18_13-16-40/2019-07-18_13-16-40/experiment1/recording1/continuous/Neuropix-PXI-100.0")
probe_ttl_dir = ''
save_dir = os.path.join(recordingPath, 'test_outputs')
if mapRf:
    stimulus_path = ".../VisualStimuli/stimuli/locally_light_sparse_noise_36_22_target_size_3_targets_per_frame_2_trials_10_background_0.0_20181120.npy"
    stimulus_data = np.load(stimulus_path, encoding='latin1', allow_pickle=True).item()
    sparse_noise_stim = stimulus_data["frames"].astype(float)  # Sparse noise stimuli with shape = (ny, nx, nframes).
    STA_lags = np.arange(-7, 3)  # The range of lags for computing STA.
    subplots_rc = (24, 16)  # (nrows, ncols) or None. The number of rows and columns for the RF subplots.
    fig_fname = "STA_RFs.png"  # The filename for saving the plotted figure.
    fig_size_pix = None  # (width, height) or None. The size of the figure (STA RFs) in pixel.


#%% Extract MUA from Neuropixels data

pix_data = mua.neuropixData(
    stim_ttl_dir,  # The folder path of the stimulus TTLs (path until .../TTL_1).
    raw_data_dir,  # The folder path of raw data (path until .../Neuropix-3a-100.0).
    save_dir,  # The folder path for saving the outputs.
    fname_extensions=(None, None),  # The labels (prefix and postfix) for the files to be saved. Postfix is useful in case there are multiple probes.
    total_ch=384,  # The total number of Neuropixels channels in use.
    event_keys=[
        (1, "locally_sparse_noise"),
        (4, "Opto"),
        (8, "Camera"),
    ],  # List containing tuples of channel states (int) and their corresponding TTL keys.
    align_to_probe_timestamps=False,  # If True, the stim TTLs in stim_ttl_dir will be aligned to NP probe timestamps (sync TTLs in probe_ttl_dir).
    stim_sync_ch=1,  # The channel state for the sync channel of the stimulus TTLs.
    probe_sync_ch=1,  # The channel state for the sync channel of the probe TTLs.
    probe_ttl_dir= probe_ttl_dir,  # The folder path of the probe TTLs.
    isSpikeGlx=False
)
"""Can do something here before extracting the MUAs, e.g., get certain TTLs 
from the saved stim_TTLs.npy as the start and stop times for the MUA extraction.
"""
pix_data.extractMua(
    spike_event_std_thresh=-3,  # The multiple of standard deviation of Butterworth bandpass filtered signals to be considered as spiking events.
    extractStartTimeSec=None,  # The start time (in second, Neuropixels time) of the data to be extracted. If None, the data will be extracted from the beginning.
    extractStopTimeSec=None,  # The end time (in second, Neuropixels time) of the data to be extracted. If None, the data will be extracted until the end.
    sliceLenSec=600,  # int, the length (in second) for slicing the data, in case the data size is too big.
    periStimExtractionKeys=['locally_sparse_noise'],   # list or None. The event/TTL keys for extracting the peri-stimulus MUA (instead of extracting everything). Any overlapping slices will be merged.
    periStimPreDurationSecs=[10],   # list of int or float. The durations (correspond to periStimExtractionKeys) in second before the stimulus onsets/TTLs for the peri-stimulus extraction.
    periStimTotDurationSecs=[20],   # list of int or float. The total durations (correspond to periStimExtractionKeys) in second for the peri-stimulus extraction.
    n_cores=int(os.cpu_count()/2)  # The number of CPUs to be used for parallel computing. Spare some CPUs so that the computer is not slowed down for other processes.
)


#%% Analysis

# =============================================================================
# Load data
# =============================================================================

stimTtlPath = os.path.join(save_dir, "stim_TTLs.npy")
spiketimePath = os.path.join(save_dir, "spiketimes.npy")
ttlDict = np.load(stimTtlPath, allow_pickle=True, encoding='latin1').item()
stDict = np.load(spiketimePath, allow_pickle=True, encoding='latin1').item()


# =============================================================================
# Compute STA and plot the RFs
# =============================================================================

if mapRf:
    LSN_frametimes = ttlDict["locally_sparse_noise"]  # Plese use the key for the stimulus frametimes/TTLs when extracting the NP MUA above.
    STA = mua.get_STA(sparse_noise_stim, stDict, LSN_frametimes, STA_lags, save_dir)
    STA_RF_fig = STA.plot(subplots_rc, fig_fname, fig_size_pix)
    RF_overview_fig = mua.plot_RF_overview(
            STA.STA_RFs, 
            sparse_noise_stim,  # Shape = (ny, nx, nframes)
            stDict, 
            LSN_frametimes,  # The timestamps of the sparse noise stimuli
            psth_start_sec=-0.1,  # the start time of the PSTH in second
            psth_end_sec=0.3,  # the stop time of the PSTH in second
            psth_interv_sec=0.01,  # the bin size of the PSTH in second
            stim_startEnd_sec=(0.1,0.3),  # the start and stop times of the stimulus in second for computing the SNR, can visually inspect the PSTH responses for better plotting of RF contours
            RF_contour_lvl=0.5, 
            target_LSN_stim=1,  # The target stimulus in the sparse-noise-stimulus matrix
            SNR_thresh=0.2,  # The SNR threshold for plotting the RF contours. The RF contours that have PSTH with SNR lower than this threshold will not be plotted.
            resp_thresh=0.3,  # The response threshold relative to the max PSTH response for plotting the RF contours
            sampling_rate=pix_data._sampling_rate,  # The sampling rate of the data acquisition devices. 30000 for Neuropixels.
            figsize=(15,10),
            psth_tick_interv=20,
    )


# =============================================================================
# Compute and plot PSTHs
# =============================================================================

samplingRate = 30000
eventsToPlot = ["locally_sparse_noise"]
psthStartSec = -1
psthEndSec = 2
psthBinSec = 0.01
psthRange = np.arange(psthStartSec, psthEndSec, psthBinSec) * samplingRate

if computePsth:
    PSTHs = []
    for e, eventKey in enumerate(eventsToPlot):
        ttls = ttlDict[eventKey]
        psths = mua.getChPsths(stDict, ttls, psthRange)
        PSTHs.append(psths)
    
    # Plot PSTH
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


