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

recordingPath = '.../2022-11-18_10-09-02'
stim_ttl_dir = os.path.join(recordingPath, "Record Node 102/experiment1/recording1/events/NI-DAQmx-100.PXIe-6341/TTL")
raw_data_dir = os.path.join(recordingPath, "Record Node 105/experiment1/recording1/continuous/Neuropix-PXI-103.ProbeA-AP")
probe_ttl_dir = os.path.join(recordingPath, "Record Node 105/experiment1/recording1/events/Neuropix-PXI-103.ProbeA-AP/TTL")
lfpTimePath = os.path.join(recordingPath, 'Record Node 105/experiment1/recording1/continuous/Neuropix-PXI-103.ProbeA-LFP/sample_numbers.npy')
lfpSyncDir = os.path.join(recordingPath, 'Record Node 105/experiment1/recording1/events/Neuropix-PXI-103.ProbeA-LFP/TTL')
lfpSyncCh = 1
toBeAlignedTimestampsInfo = [('LFP_times', lfpTimePath, None, lfpSyncDir, lfpSyncCh), ]   # [(key1, timestampsPath1, statesOrInfoPath1, syncDirPath1, syncCh1), ...]
save_dir = recordingPath
nTotCh = 384


#%% Extract MUA from Neuropixels data

pix_data = mua.neuropixData(
    stim_ttl_dir,  # The folder path of the stimulus TTLs (path until .../TTL_1).
    raw_data_dir,  # The folder path of raw data (path until .../Neuropix-3a-100.0).
    save_dir,  # The folder path for saving the outputs.
    fname_extensions=(None, None),  # The labels (prefix and postfix) for the files to be saved. Postfix is useful in case there are multiple probes.
    total_ch=nTotCh,  # The total number of Neuropixels channels in use.
    event_keys=[
        (1, "Sync"),
        (2, "Opto"),
        (4, "Camera"),
    ],  # List containing tuples of channel states (int) and their corresponding TTL keys.
    align_to_probe_timestamps=True,  # If True, the stim TTLs in stim_ttl_dir will be aligned to NP probe timestamps (sync TTLs in probe_ttl_dir).
    stim_sync_ch=1,  # The channel state for the sync channel of the stimulus TTLs.
    probe_sync_ch=1,  # The channel state for the sync channel of the probe TTLs.
    probe_ttl_dir=probe_ttl_dir,  # The folder path of the probe TTLs.
    toBeAlignedTimestampsInfo=toBeAlignedTimestampsInfo,  # Timestamps (besides stimulus TTLs) to be aligned to probe time. Format in a list of tuples, e.g., [(key1, timestampsPath1, statesOrInfoPath1), ...]
)
pix_data.extractMua(
    spike_event_std_thresh=-3,  # The multiple of standard deviation of Butterworth bandpass filtered signals to be considered as spiking events.
    extractStartTimeSec=None,  # The start time (in second, Neuropixels time) of the data to be extracted. If None, the data will be extracted from the beginning.
    extractStopTimeSec=None,  # The end time (in second, Neuropixels time) of the data to be extracted. If None, the data will be extracted until the end.
    sliceLenSec=600,  # int, the length (in second) for slicing the data, in case the data size is too big.
    periStimExtractionKeys=None,   # list or None. The event/TTL keys for extracting the peri-stimulus MUA (instead of extracting everything). Any overlapping slices will be merged.
    periStimPreDurationSecs=[10],   # list of int or float. The durations (correspond to periStimExtractionKeys) in second before the stimulus onsets/TTLs for the peri-stimulus extraction.
    periStimTotDurationSecs=[20],   # list of int or float. The total durations (correspond to periStimExtractionKeys) in second for the peri-stimulus extraction.
    n_cores=int(os.cpu_count()/2)  # The number of CPUs to be used for parallel computing. Spare some CPUs so that the computer is not slowed down for other processes.
)


#%% Load and analyze data

# =============================================================================
# Parameters
# =============================================================================

samplingRateAP = 30000
samplingRateLFP = 2500
bitVolts = 0.195   # microvolts per bit, from structure.oebin ("bit_volts")
lfpPath = os.path.join(recordingPath, "Record Node 105/experiment1/recording1/continuous/Neuropix-PXI-103.ProbeA-LFP/continuous.dat")
lfpStimKey = 'Opto'
lfpPreStimSec = 1
lfpTrialDurationSec = 5


# =============================================================================
# Load data
# =============================================================================

stimTtlPath = os.path.join(save_dir, "stim_TTLs.npy")
spiketimePath = os.path.join(save_dir, "spiketimes.npy")
stimTtlDict = np.load(stimTtlPath, allow_pickle=True, encoding='latin1').item()
stDict = np.load(spiketimePath, allow_pickle=True, encoding='latin1').item()
lfpData = np.memmap(lfpPath, dtype=np.int16, mode="r").reshape((-1, nTotCh)).T
lfpMicroVs = lfpData * bitVolts
lfpPreStim = int(round(lfpPreStimSec * samplingRateLFP))
lfpTrialDuration = int(round(lfpTrialDurationSec * samplingRateLFP))
lfpStimTimes = stimTtlDict[lfpStimKey]
lfpTimes = stimTtlDict['LFP_times']['alignedTimes']


# =============================================================================
# Compute average channel firing rate
# =============================================================================

allSpktimes = np.hstack(list(stDict.values()))
durationSec = (allSpktimes.max() - allSpktimes.min()) / samplingRateAP
nSpks = np.array([len(stCh) for stCh in stDict.values()])
avgFiringRates = nSpks / durationSec


# =============================================================================
# Compute stimulus-triggered averaged LFP and CSD
# =============================================================================

stimLfps = []
minLen = lfpTrialDuration
for stimTime in lfpStimTimes:
    start = stimTime - lfpPreStim
    end = start + lfpTrialDuration
    mask = (lfpTimes >= start) & (lfpTimes < end)
    stimLfps.append(lfpMicroVs[:, mask])
    nTime = mask.sum()
    minLen = nTime if nTime < minLen else minLen
stimAvgLfp = np.mean([lfp[:,:minLen] for lfp in stimLfps], axis=0)
csd = mua.getCurrentSourceDensity(
    stimAvgLfp, samplingIntervalUm=100, nInterv=2, chSeparationUm=10, mode='same')


#%% Plotting

interv = np.diff(lfpTimes).mean()
timeTicks = np.round(np.arange(0, lfpTrialDuration+1, samplingRateLFP) / interv).astype(int)
timesSec = np.arange(-lfpPreStimSec, lfpTrialDurationSec, 1)
fig, axs = plt.subplots(1, 3, figsize=(15, 3))

plt.sca(axs[0])
plt.plot(avgFiringRates, c='k')
plt.xlabel('Channel')
plt.ylabel('Average firing rate (Hz)')
plt.title('Average firing rate for each channel')

plt.sca(axs[1])
plt.imshow(stimAvgLfp)
plt.xticks(timeTicks, timesSec)
cbar = plt.colorbar()
cbar.ax.set_title('μV')
plt.xlabel('Time (s)')
plt.ylabel('Channel')
plt.title('Stimulus-triggered averaged LFP')

plt.sca(axs[2])
plt.imshow(csd)
plt.xticks(timeTicks, timesSec)
cbar = plt.colorbar()
cbar.ax.set_title('μV / μm$^2$')
plt.xlabel('Time (s)')
plt.ylabel('Valid channel')
plt.title('Current source density')

plt.tight_layout()


