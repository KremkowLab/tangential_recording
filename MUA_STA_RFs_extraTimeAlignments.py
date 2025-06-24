#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:29:51 2021

@author: kailun
"""

import os
import MUA_STA_RFs as mua


#%% Parameters

recordingPath = '.../rf_mapping_LIN-20250623T142337Z-1-001/rf_mapping_LIN'
stim_ttl_dir = os.path.join(recordingPath, "NI-DAQmx-102.PXIe-6341/TTL")
raw_data_dir = os.path.join(recordingPath, "2019-07-18_13-16-40/2019-07-18_13-16-40/experiment1/recording1/continuous/Neuropix-PXI-100.0")
probe_ttl_dir = os.path.join(recordingPath, 'Neuropix/TTL-20250623T144951Z-1-001/TTL')
timePath = os.path.join(recordingPath, 'MessageCenter/sample_numbers.npy')
infoPath = os.path.join(recordingPath, 'MessageCenter/text.npy')
toBeAlignedTimestampsInfo = [('MessageCenter', timePath, infoPath), ]   # [(key1, timestampsPath1, statesOrInfoPath1), ...]
save_dir = os.path.join(recordingPath, 'test_outputs')


#%% Extract MUA from Neuropixels data

pix_data = mua.neuropixData(
    stim_ttl_dir,  # The folder path of the stimulus TTLs (path until .../TTL_1).
    raw_data_dir,  # The folder path of raw data (path until .../Neuropix-3a-100.0).
    save_dir,  # The folder path for saving the outputs.
    fname_extensions=(None, None),  # The labels (prefix and postfix) for the files to be saved. Postfix is useful in case there are multiple probes.
    total_ch=384,  # The total number of Neuropixels channels in use.
    event_keys=[
        (1, "sync"),
        (2, "ttls"),
    ],  # List containing tuples of channel states (int) and their corresponding TTL keys.
    align_to_probe_timestamps=True,  # If True, the stim TTLs in stim_ttl_dir will be aligned to NP probe timestamps (sync TTLs in probe_ttl_dir).
    stim_sync_ch=1,  # The channel state for the sync channel of the stimulus TTLs.
    probe_sync_ch=1,  # The channel state for the sync channel of the probe TTLs.
    probe_ttl_dir=probe_ttl_dir,  # The folder path of the probe TTLs.
    chStep=1,  # The step in taking the channels to be extracted.
    toBeAlignedTimestampsInfo=toBeAlignedTimestampsInfo,  # Timestamps (besides stimulus TTLs) to be aligned to probe time. Format in a list of tuples, e.g., [(key1, timestampsPath1, statesOrInfoPath1), ...]
)
pix_data.extractMua(
    spike_event_std_thresh=-4,  # The multiple of standard deviation of Butterworth bandpass filtered signals to be considered as spiking events.
    extractStartTimeSec=None,  # The start time (in second, Neuropixels time) of the data to be extracted. If None, the data will be extracted from the beginning.
    extractStopTimeSec=None,  # The end time (in second, Neuropixels time) of the data to be extracted. If None, the data will be extracted until the end.
    sliceLenSec=None,  # int, the length (in second) for slicing the data, in case the data size is too big.
    periStimExtractionKeys=None,   # list or None. The event/TTL keys for extracting the peri-stimulus MUA (instead of extracting everything). Any overlapping slices will be merged.
    periStimPreDurationSecs=[10],   # list of int or float. The durations (correspond to periStimExtractionKeys) in second before the stimulus onsets/TTLs for the peri-stimulus extraction.
    periStimTotDurationSecs=[20],   # list of int or float. The total durations (correspond to periStimExtractionKeys) in second for the peri-stimulus extraction.
    n_cores=int(os.cpu_count()/2)  # The number of CPUs to be used for parallel computing. Spare some CPUs so that the computer is not slowed down for other processes.
)


