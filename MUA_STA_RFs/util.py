#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:12:16 2019

@author: kailun
"""

import numpy as np
from scipy.signal import butter, filtfilt
from scipy import interpolate
from platform import python_version as py_ver
import os, shutil
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


# =============================================================================
# Signal processing
# =============================================================================

def butter_bandpass(low_cut, high_cut, sampling_rate, order=2):
    """Butterworth bandpass filter from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    PARAMETERS
    -----------
    low_cut : int or float
        The low of the passband/frequency range.
    high_cut : int or float
        The high of the passband/frequency range.
    sampling_rate : int or float
        The data sampling rate.
    order : int
        The order of the filter.
    
    RETURN
    ------
    bandpassed_data : tuple
        Tuple of a and b. Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyq = 0.5 * sampling_rate
    low = low_cut / nyq
    high = high_cut / nyq
    bandpassed_data = butter(order, [low, high], btype="band")
    return bandpassed_data

def butter_bandpass_filter(data, low_cut, high_cut, sampling_rate, order=2):
    """To bandpass filter the given data with Butterworth filter.
    PARAMETERS
    ----------
    data : array-like, 1d
        The signals to be filtered.
    low_cut : int or float
        The low of the passband/frequency range.
    high_cut : int or float
        The high of the passband/frequency range.
    sampling_rate : int or float
        The data sampling rate.
    order : int
        The order of the filter.
    
    RETURN
    ------
    filt_data : array-like, 1d
        The bandpass filtered data.
    """
    b, a = butter_bandpass(low_cut, high_cut, sampling_rate, order)
    filt_data = filtfilt(b, a, data)
    return filt_data

def basic_peak_detector(sig, orientation="negative", thresh=-3.5, verbose=False):
    """To detect spiking events.
    PARAMETERS
    ----------
    sig : array-like, 1d
        The signal to be processed.
    orientation : str
        "Positive", "negative", or "both". The side of the signals to be considered.
    thresh : int or float
        The threshold of the signals to be considered as spiking events.
    verbose : bool
        If True, the details of the spike event detection will be shown.
    
    RETURN
    ------
    spike_event_ind : array-like, 1d
        The index (spiketimes) where spike events occur.
    """
    orientation = orientation.lower()
    if orientation.lower() != "positive":
        sig0_neg = sig[0:-2]
        sig1_neg = sig[1:-1]
        sig2_neg = sig[2:]
        (peak_ind_neg,) = np.nonzero(
            (sig1_neg <= sig0_neg) & (sig1_neg < sig2_neg) & (sig1_neg < thresh)
        )
    if orientation.lower() != "negative":
        sig_inv = sig * -1
        sig0_pos = sig_inv[0:-2]
        sig1_pos = sig_inv[1:-1]
        sig2_pos = sig_inv[2:]
        (peak_ind_pos,) = np.nonzero(
            (sig1_pos <= sig0_pos) & (sig1_pos < sig2_pos) & (sig1_pos < thresh)
        )
    if orientation.lower() == "both":
        peak_ind_neg = np.array([peak_ind_neg])
        peak_ind_pos = np.array([peak_ind_pos])
        peak_ind = np.concatenate((peak_ind_neg.ravel(), peak_ind_pos.ravel()), axis=0)
        peak_ind = np.sort(peak_ind)
    ind = (
        peak_ind
        if orientation.lower() == "both"
        else peak_ind_pos
        if orientation.lower() == "positive"
        else peak_ind_neg
    )
    if verbose:
        size = len(sig)
        n = len(ind)
        print("nb peak={}// {}% of datapints over thr".format(n, n / size * 100))
    spike_event_ind = ind + 1
    return spike_event_ind

def filter_detect(
    channel_data,
    start,
    thresh,
    orientation,
    low_cut,
    high_cut,
    sampling_rate,
    order=2,
    verbose=False,
):
    """To bandpass filter the given data and detect spiking events.
    PARAMETERS
    ----------
    channel_data : array-like, 1d
        The data of one Neuropixels channel to be filtered.
    start : int
        The start time of the data in unit time.
    thresh : int or float
        The number of standard deviation of Butterworth bandpass filtered
        signals to be considered as spiking events.
    orientation : str
        "Positive", "negative", or "both". The side of the signals to be considered.
    low_cut : int or float
        The low of the passband/frequency range.
    high_cut : int or float
        The high of the passband/frequency range.
    sampling_rate : int or float
        The data sampling rate.
    order : int
        The order of the filter.
    verbose : bool
        If True, the details of the spike event detection will be shown.
    
    RETURN
    ------
    spiketimes_tmp : array-like, 1d
        The detected spiketimes from the data.
    """
    data_filt = butter_bandpass_filter(channel_data, low_cut, high_cut, sampling_rate, order)
    threshold = thresh * np.std(data_filt)
    spiketimes_tmp = basic_peak_detector(data_filt, orientation, threshold, verbose)
    spiketimes_tmp += start
    return spiketimes_tmp


# =============================================================================
# Timestamp alignment
# =============================================================================

def align_and_save_timestamps(
    to_be_aligned_sync_ch, 
    ref_sync_ch, 
    to_be_aligned_ttl_dir, 
    ref_ttl_dir, 
    chState_fname, 
    unitTimestamps_fname
):
    """To align the given timestamps to the referece timestamps.
    PARAMETERS
    ----------
    to_be_aligned_sync_ch : int
        The channel state of the sync TTLs to be aligned.
    ref_sync_ch : int
        The channel state of the reference sync TTLs.
    to_be_aligned_ttl_dir : str
        The path to folder containing the TTLs/timestamps to be aligned. 
        E.g. the NIDAQ TTL folder (path until .../NI-DAQmx-102.0/TTL_1).
    ref_ttl_dir : str
        The path to folder containing the TTLs/timestamps as reference.
        E.g. the Neuropixel TTL folder (path until .../Neuropix-PXI-100.0/TTL_1).
    chState_fname, unitTimestamps_fname : str
        The filenames for channel states and timestamps in unit time/sample numbers.

    RETURN
    ------
    new_ttl_dir : str
        The path to the folder containing aligned TTLs/timestamps for TTLs
        in to_be_aligned_ttl_dir.
    """
    to_be_aligned_chState_fpath = os.path.join(to_be_aligned_ttl_dir, chState_fname)
    to_be_aligned_timestamps_fpath = os.path.join(to_be_aligned_ttl_dir, unitTimestamps_fname)
    ref_chState_fpath = os.path.join(ref_ttl_dir, chState_fname)
    ref_timestamps_fpath = os.path.join(ref_ttl_dir, unitTimestamps_fname)
    timestamps_sync, timestamps = get_timestamps(
        to_be_aligned_timestamps_fpath, to_be_aligned_chState_fpath, to_be_aligned_sync_ch
    )
    reference_sync, timestamps_ref = get_timestamps(
        ref_timestamps_fpath, ref_chState_fpath, ref_sync_ch
    )
    new_ttl_dir = to_be_aligned_ttl_dir + "_aligned"
    if not os.path.exists(new_ttl_dir):
        os.makedirs(new_ttl_dir)
    timestamps_aligned = align_timestamps(timestamps, timestamps_sync, reference_sync)
    np.save(os.path.join(new_ttl_dir, unitTimestamps_fname), timestamps_aligned)
    shutil.copy(os.path.join(to_be_aligned_ttl_dir, chState_fname), new_ttl_dir)
    return new_ttl_dir

def align_timestamps(timestamps, timestamps_sync, reference_sync, verbose=False):
    """To align given TTLs/timestamps by aligning its sync TTLs to the reference sync TTLs.
    E.g. align NIDAQ timestamps to Neuropixels probe (NP) timestamps.
    PARAMETERS
    ----------
    timestamps : array-like, 1d
        The TTLS/timestamps to be aligned, e.g. the NIDAQ TTL timestamps.
    timestamp_sync : array-like, 1d
        The sync TTLs of the timestamps to be aligned, e.g. NIDAQ sync.
    reference_sync : array-like, 1d
        The reference sync timestamps, e.g. NP sync.
    verbose : bool
        If True, the user will be notified if the timestamps_sync and the 
        reference_sync do not have the same length.
    
    RETURN
    ------
    timestamps_aligned : array-like, 1d
        The aligned timestamps for the given timestamps.
    """
    if len(timestamps_sync) != len(reference_sync):
        if verbose:
            print(
                "The timestamps_sync ({}) and reference_sync ({}) do not have same length.".format(
                    len(timestamps_sync), len(reference_sync)
                )
            )
        min_len = min(len(timestamps_sync), len(reference_sync))
        timestamps_sync = timestamps_sync[:min_len]
        reference_sync = reference_sync[:min_len]
    dt = timestamps_sync - reference_sync
    func_dt = interpolate.interp1d(timestamps_sync, dt, fill_value="extrapolate")
    timestamps_aligned = timestamps - func_dt(timestamps)
    if timestamps.dtype in [int, np.int64]:
        timestamps_aligned = np.round(timestamps_aligned).astype(np.int64)
    return timestamps_aligned

def get_timestamps(timestamps_fpath, chState_fpath, target_channel):
    """To get the target timestamps given its channel state.
    PARAMETERS
    ----------
    timestamps_fpath, chState_fpath : str
        The filepaths for timestamps in unit time/sample numbers and channel states.
    target_channel : int
        The channel state of the target channel.
    
    RETURN
    ------
    target_timestamps : array-like, 1d
        The timestamps of the target_channel.
    all_timestamps : array-like, 1d
        All timestamps in the ttl_dir.
    """
    all_timestamps = np.load(timestamps_fpath)
    channel_states = np.load(chState_fpath)
    target_timestamps = all_timestamps[channel_states == target_channel]
    return target_timestamps, all_timestamps


# =============================================================================
# Analysis and plotting
# =============================================================================

def plot_RF_overview(
    RFs, 
    stimulus, 
    spiketimes, 
    frametimes, 
    psth_start_sec, 
    psth_end_sec, 
    psth_interv_sec,
    stim_startEnd_sec,
    RF_contour_lvl, 
    target_LSN_stim=1,
    SNR_thresh=0.,
    resp_thresh=0.,
    sampling_rate=30000,
    figsize=(10,5),
    psth_tick_interv=10,
):
    """To plot the PSTH of the max STA RF pixel and the RF contours for all channels.
    PARAMETERS
    ----------
    RFs : array-like, 3D
        The STA receptve fields for all channels. Shape = (n_chs, ylen, xlen).
    stimulus : array-like, 3d
        Locally sparse noise stimuli with shape = (ny, nx, nframes).
    spiketimes : dict
        Dictionary containing the spiketimes for each channel.
        The dictionary keys are the channels.
    frametimes : array_like, 1d
        The stimulus timestamps/TTLs for the locally sparse noise frames.
    psth_start_sec, psth_end_sec, psth_interv_sec : float
        The start, end, and interval for the PSTH bins in second.
    stim_startEnd_sec : tuple or list
        The start and end of a stimulus frame in second.
    RF_contour_lvl : float
        The RF contour level to be plotted.
    target_LSN_stim : int or float
        The target stimulus in the sparse-noise-stimulus matrix.
    SNR_thresh, resp_thresh : float
        The SNR and response thresholds for plotting the RF contours.
    sampling_rate : int or float
        The Neuropixels' sampling rate in Hz.
    psth_tick_interv : int
        The interval for labeling the PSTH bins.
    
    RETURN
    ------
    fig : matplotlib object
        The figure containing the RFs' PSTH and RF contours for all channels.
    """
    psth_range_sec, psth_range = getPsthRange(
        psth_start_sec, psth_end_sec, psth_interv_sec, sampling_rate)
    maxRFpsths = get_maxRFpixel_psth(RFs, stimulus, spiketimes, frametimes, 
                                     psth_range, target_LSN_stim)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1 = plotChPsths(ax1, maxRFpsths, psth_range, psth_range_sec, psth_tick_interv)
    ax1.set_title("Max RF pixel PSTHs")
    
    plt.sca(ax2)
    stim_mask = get_stim_mask(psth_range[:-1], np.array(stim_startEnd_sec)*sampling_rate)
    SNR = get_PSTH_SNR(maxRFpsths, stim_mask)
    for r, RF in enumerate(RFs):
        has_high_SNR = SNR[r] >= SNR_thresh
        has_decent_response = maxRFpsths[r].max() >= (resp_thresh*maxRFpsths.max())
        if has_high_SNR and has_decent_response:
            plt.contour(RF, [RF_contour_lvl])
    ax2.set_aspect("equal")
    ax2.set_title("RF contours")
    return fig

def plotChPsths(
        ax, psths, psth_range, psth_range_sec, psth_tick_interv=10, 
        vmin=None, vmax=None, xtickRotation=25):
    """Plot PSTHs for all channels.
    
    PARAMETERS
    ----------
    psths : array-like, 2D
        The PSTHs to be plotted. Shape = (nCh, psthLen).
    psth_tick_interv : int
        The interval for labeling the PSTH bins.
    """
    ax.imshow(psths, cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(0, psth_range.shape[0], int(psth_tick_interv)))
    ax.set_xticklabels(
        np.round(psth_range_sec, 2)[::int(psth_tick_interv)], rotation=xtickRotation)
    ax.set_aspect("equal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    return ax

def plotNpix2ChMappedPsths(
        chMappedPsths, psth_range, psth_range_sec, psth_tick_interv=10, 
        sameClim=True, xtickRotation=25, figsize=(10,10)):
    """
    chMappedPsths : array-like, 3D
        The Neuropixels 2 channel mapped PSTHs. Shape = (nShank, nCh, psthLen).
    """
    nShank, nCh, psthLen = chMappedPsths.shape
    clim = [None, None]
    if sameClim:
        mnVal = np.nanmin(chMappedPsths)
        mxVal = np.nanmax(chMappedPsths)
        clim = [mnVal, mxVal]
    fig, axs = plt.subplots(1, nShank, figsize=figsize)
    for i, ax in enumerate(axs):
        ax = plotChPsths(
            ax, chMappedPsths[i], psth_range, psth_range_sec, psth_tick_interv, 
            *clim, xtickRotation)
        ax.set_title(f'Shank {i+1}')
    return fig

def getPsthRange(psth_start_sec, psth_end_sec, psth_interv_sec, sampling_rate):
    psth_range_sec = np.arange(psth_start_sec, psth_end_sec, psth_interv_sec)
    psth_range_ut = psth_range_sec * sampling_rate
    return psth_range_sec, psth_range_ut

def get_PSTH_SNR(PSTHs, stim_mask):
    """To compute the SNR for a list of PSTHs given a stimulus mask.
    PARAMETERS
    ----------
    PSTHs : array-like, 2D
        The PSTHs to be used. Shape = (nCh, n_psth_bins).
    stim_mask : array-like, 1D
        The boolean mask indicating the stimulus time points/bins.
    
    RETURN
    ------
    SNR_norm : array-like, 1D
        The normalized SNR for each PSTH.
    """
    signal = (PSTHs*stim_mask).sum(-1) / stim_mask.sum()
    noise = (PSTHs*~stim_mask).sum(-1) / (~stim_mask).sum()
    if sum(noise==0) > 0:
        min_noise = PSTHs.mean() / 10
        PSTHs_tmp = PSTHs + min_noise
        signal = (PSTHs_tmp*stim_mask).sum(-1) / stim_mask.sum()
        noise = (PSTHs_tmp*~stim_mask).sum(-1) / (~stim_mask).sum()
    SNR = signal / noise
    SNR_norm = SNR / SNR.max()
    return SNR_norm

def get_stim_mask(time_points, stim_startEnd):
    """To get the mask for stimulus given a time series, the stimulus duration 
    and the time points given have to be in same unit.
    PARAMETERS
    ----------
    time_points : array-like, 1D
        The series of time points of interest.
    stim_startEnd : tuple or list
        The stimulus start and end times with unit same as time_points.
    
    RETURN
    ------
    stim_mask : array-like, 1D
        The boolean mask indicating the stimulus time points.
    """
    stim_start, stim_end = stim_startEnd
    stim_mask = (time_points>=stim_start) & (time_points<stim_end)
    return stim_mask

def get_maxRFpixel_psth(RFs, stimulus, spiketimes, frametimes, psth_range, 
                        target_stim=1):
    """To compute the PSTH of the max STA RF pixel for all channels.
    PARAMETERS
    ----------
    RFs : array-like, 3D
        The STA receptve fields for all channels. Shape = (n_chs, ylen, xlen).
    stimulus : array-like, 3d
        Locally sparse noise stimuli with shape = (ny, nx, nframes).
    spiketimes : dict
        Dictionary containing the spiketimes for each channel.
        The dictionary keys are the channels.
    frametimes : array_like, 1d
        The stimulus timestamps/TTLs for the locally sparse noise frames.
    psth_range : array-like, 1D
        The bins for computing the PSTH. Length = n_psth_bins.
    target_stim : int or float
        The target stimulus in the stimulus matrix.
    
    RETURN
    ------
    psths : array-like, 2D
        The computed PSTHs for all channels. Shape = (n_chs, n_psth_bins).
    """
    st_keys = list(spiketimes.keys())
    psths = np.zeros((RFs.shape[0], psth_range.shape[0]-1))
    for ch, RF in enumerate(RFs):
        idx = np.argmax(np.abs(RF))
        y, x = np.unravel_index(idx, RF.shape)
        triggers, = np.where(stimulus[y, x, :]==target_stim)
        ft = frametimes[triggers]
        st = spiketimes[st_keys[ch]]
        psths[ch] = calc_psth(st, ft, psth_range)
    return psths

def getChPsths(spiketimes, ttls, psth_range, n_cores=os.cpu_count()/2):
    """To compute the PSTHs for all channels given TTL triggers.
    
    PARAMETERS
    ----------
    spiketimes : dict
        Dictionary containing the spiketimes for each channel.
        The dictionary keys are the channels.
    """
    n_cores = int(n_cores)
    chs = list(spiketimes.keys())
    n_psth = psth_range.shape[0] - 1
    if n_cores > 1:
        psths = (Parallel(n_jobs=n_cores, verbose=10)
                 (delayed(calc_psth)
                 (spiketimes[ch], ttls, psth_range) for ch in chs))
        psths = np.array(psths).reshape(len(chs), n_psth)
    else:
        psths = np.zeros((len(chs), n_psth))
        for c, ch in enumerate(chs):
            st = spiketimes[ch]
            psths[c] = calc_psth(st, ttls, psth_range)
    return psths

def calc_psth(spiketimes, frametimes, psth_range):
    """To compute the PSTH.
    PARAMETERS
    ----------
    spiketimes : array_like, 1d
        The spiketimes of a channel/cluster.
    frametimes : array_like, 1d
        The stimulus timestamps/TTLs for the locally sparse noise frames.
    psth_range : array-like, 1D
        The bins for computing the PSTH. Length = n_psth_bins.
        
    RETURN
    ------
    psth : array-like, 1D
        The computed PSTH. Length = n_psth_bins.
    """
    psth_tmp = np.zeros(psth_range.shape[0] - 1)
    for ft in frametimes:
        freq, x_axis = np.histogram(spiketimes - ft, psth_range)
        psth_tmp += freq
    psth = psth_tmp / len(frametimes)
    return psth

def mapPsthsToShank(psths, chMap, emptyVal=np.nan):
    """
    Parameters
    ----------
    psths : array-like, 2D
        Shape = (nCh, psthLen).
    chMap : array-like, 2D
        Channel mapping. Shape = (nCh, 5).
    
    Return
    ------
    chMappedPsths : array-like, 3D
        Shape = (nShank, nCh, psthLen).
    """
    nCh, psthLen = psths.shape
    shankInd = np.unique(chMap[:, 1])
    nShank = len(shankInd)
    chMappedPsths = np.full((nShank, nCh, psthLen), emptyVal)
    for i, shankIdx in enumerate(shankInd):
        shankMask = chMap[:, 1] == shankIdx
        shankChInd = chMap[:, -1][shankMask]
        shankIdxArr = np.array([shankIdx] * len(shankChInd))
        chMappedPsths[shankIdxArr, shankChInd] = psths[shankMask]
    return chMappedPsths


# =============================================================================
# Miscellaneous
# =============================================================================

def get_action(message):
    """To get the action for a situation whether to stop or continue (ignore the concern).
    PARAMETERS
    ----------
    message : str
        The message to be shown when certain situation happened.
    """
    print(message)
    if py_ver()[0] == "2":
        action = raw_input("Action (pass, stop): ")
    elif py_ver()[0] == "3":
        action = input("Action (pass, stop): ")
    while action != "stop" and action != "pass":
        print(message)
        if py_ver()[0] == "2":
            action = raw_input("Action (pass, stop): ")
        elif py_ver()[0] == "3":
            action = input("Action (pass, stop): ")
    if action == "pass":
        pass
    elif action == "stop":
        raise KeyboardInterrupt("Extraction terminated!")

def get_rawdata_timestamps_info(timestamps):
    """
    PARAMETERS
    ----------
    timestamps : array-like, 1d
        The timestamps of the raw data.
    
    RETURN
    ------
    start_time : int
        The start time of the raw data recording.
    total_timestamp_len : array-like, 1d
        The length of the data recording in unit time.
    """
    timestamps_len = timestamps.shape[0]
    start_time = timestamps[0]
    time_diff = np.diff(timestamps)
    (loss_time_idx,) = np.where(time_diff != 1)
    loss_time_len = time_diff[loss_time_idx].sum() - loss_time_idx.size
    total_timestamp_len = timestamps_len + loss_time_len
    return start_time, total_timestamp_len

def getLatestFilePath(folder, files):
    """
    files : List of the files from the latest to the oldest.
    """
    latestPath = None
    for file in files:
        curPath = os.path.join(folder, file)
        if os.path.exists(curPath):
            latestPath = curPath
            break
    if latestPath is None:
        raise ValueError(f"No matching file in the folder {folder}.")
    return latestPath

def mergeOverlappedTrials(starts, stops, maxDuration=None):
    intervals = sorted(zip(starts, stops))
    mergedIntervals = []
    for curStart, curStop in intervals:
        hasMerged = len(mergedIntervals) > 0
        if hasMerged:
            prevStart, prevStop = mergedIntervals[-1]
            noOverlap = prevStop < curStart
        if not hasMerged or noOverlap:
            mergedIntervals.append((curStart, curStop))
        else:
            mergedIntervals[-1] = (prevStart, max(prevStop, curStop))
    if maxDuration is not None:
        mergedIntervals = splitTrials(mergedIntervals, maxDuration)
    starts, stops = zip(*mergedIntervals)
    return np.array(starts), np.array(stops)

def splitTrials(trials, maxDuration):
    splittedTrials = []
    for start, stop in trials:
        duration = stop - start
        if duration <= maxDuration:
            splittedTrials.append((start, stop))
        else:
            curStart = start
            while curStart < stop:
                curStop = min(curStart + maxDuration, stop)
                splittedTrials.append((curStart, curStop))
                curStart = curStop
    return splittedTrials

def mergeOpenEphysBinaryFiles(
        saveDir, dataDirs, dataFilename='continuous.dat', 
        timeFilename='timestamps.npy', nChs=384):
    mergedFilePath = os.path.join(saveDir, dataFilename)
    mergedTimePath = os.path.join(saveDir, timeFilename)
    timeShiftPath = os.path.join(saveDir, 'timeshifts.npy')

    mergedTimes = []
    timeShifts = []
    dataLens = []
    for d, dataDir in enumerate(dataDirs):
        dataPath = os.path.join(dataDir, dataFilename)
        timePath = os.path.join(dataDir, timeFilename)
        data = np.memmap(dataPath, dtype='int16', mode='r')
        timestamps = np.load(timePath, allow_pickle=True, encoding='latin1')
        timeShift = timestamps[0] - mergedTimes[-1] - 1 if d > 0 else 0
        timeShifts = np.append(timeShifts, timeShift).astype(int)
        shiftedTimes = timestamps - timeShift
        mergedTimes = np.append(mergedTimes, shiftedTimes).astype(int)
        dataLens = np.append(dataLens, len(timestamps))
        mode = 'wb' if d == 0 else 'a'   # wb: Write data to new file; a: Append data from other recordings to the same file
        file = open(mergedFilePath, mode)
        data.tofile(file)
        file.close()
    del data
    timeShiftInfo = {}
    timeShiftInfo['timeShifts'] = timeShifts
    timeShiftInfo['timeShiftStarts'] = np.append(0, np.cumsum(dataLens[:-1])).astype(int)
    np.save(mergedTimePath, mergedTimes)
    np.save(timeShiftPath, timeShiftInfo)
    print(f'\nThe merged data is saved at {mergedFilePath}')
    print(f'The merged timestamps is saved at {mergedTimePath}')
    print(f'The merged data time shifting info is saved at {timeShiftPath}')
    return mergedTimes, timeShiftInfo


# =============================================================================
# SpikeGLX util
# =============================================================================

def saveCsvToNeuropixTimestampsFormat(
        timesSecCsvDir, prefix='ttl_', samplingRate=30000):
    """To convert timestamps from second in CSV to unit time in Neuropixels 
    format.
    """
    files = os.listdir(timesSecCsvDir)
    channelStates = []
    timestampsUt = []
    for f, file in enumerate(files):
        filePath = os.path.join(timesSecCsvDir, file)
        if not os.path.isfile(filePath):
            continue
        fname, ext = os.path.splitext(file)    
        if file.startswith(prefix) and ext == '.csv':
            idx = fname.find(prefix) + len(prefix)
            ch = int(fname[idx:])
            ttlPath = os.path.join(timesSecCsvDir, file)
            timesSec = np.genfromtxt(ttlPath, delimiter=',')
            timesUnitTime = np.round(timesSec * samplingRate).astype(int)
            channelStates = np.append(channelStates, [ch] * len(timesUnitTime))
            timestampsUt = np.append(timestampsUt, timesUnitTime)
    np.save(os.path.join(timesSecCsvDir, 'channel_states.npy'), channelStates)
    np.save(os.path.join(timesSecCsvDir, 'timestamps.npy'), timestampsUt)

def genDataTimestamps(rawDataPath, totalCh):
    rawDataDir, dataFile = os.path.split(rawDataPath)
    rawData = np.memmap(rawDataPath, dtype=np.int16, mode="r")
    rawDataLen = int(round(len(rawData) / totalCh))
    dataTimestamps = np.arange(rawDataLen)
    np.save(os.path.join(rawDataDir, 'timestamps.npy'), dataTimestamps)
    del rawData

def getNpix2ChannelMap(filePath):
    """
    filePath : str
        The path to the NP2.0_surface_2_use.imro file.
        
    Return
    ------
    dataArr : array-like, 2D
        The channel mapping, Shape = (nCh, 5).
    """
    with open(filePath, 'r') as file:
        data = file.read().strip()
    data = data.split(')', 1)[1].strip()   # Remove the first part (2013,384)
    chunks = data.split(')(')
    chunks = [chunk.replace('(', '').replace(')', '') for chunk in chunks]
    dataArr = np.array([list(map(int, chunk.split())) for chunk in chunks])
    return dataArr


