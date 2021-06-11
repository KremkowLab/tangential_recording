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
        The length of the data recoring in unit time.
    """
    timestamps_len = timestamps.shape[0]
    start_time = timestamps[0]
    time_diff = np.diff(timestamps)
    (loss_time_idx,) = np.where(time_diff != 1)
    loss_time_len = time_diff[loss_time_idx].sum() - loss_time_idx.size
    total_timestamp_len = timestamps_len + loss_time_len
    return start_time, total_timestamp_len


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
    if orientation != "positive":
        sig0_neg = sig[0:-2]
        sig1_neg = sig[1:-1]
        sig2_neg = sig[2:]
        (peak_ind_neg,) = np.nonzero(
            (sig1_neg <= sig0_neg) & (sig1_neg < sig2_neg) & (sig1_neg < thresh)
        )
    if orientation != "negative":
        sig_inv = sig * -1
        sig0_pos = sig_inv[0:-2]
        sig1_pos = sig_inv[1:-1]
        sig2_pos = sig_inv[2:]
        (peak_ind_pos,) = np.nonzero(
            (sig1_pos <= sig0_pos) & (sig1_pos < sig2_pos) & (sig1_pos < thresh)
        )
    if orientation == "both":
        peak_ind_neg = np.array([peak_ind_neg])
        peak_ind_pos = np.array([peak_ind_pos])
        peak_ind = np.concatenate((peak_ind_neg.ravel(), peak_ind_pos.ravel()), axis=0)
        peak_ind = np.sort(peak_ind)
    ind = (
        peak_ind
        if orientation == "both"
        else peak_ind_pos
        if orientation == "positive"
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


def get_timestamps(ttl_dir, target_channel):
    """To get the target timestamps given its channel state.
    PARAMETERS
    ----------
    ttl_dir : str
        The path to the folder containing the TTLs.
    target_channel : int
        The channel state of the target channel.
    
    RETURN
    ------
    target_timestamps : array-like, 1d
        The timestamps of the target_channel.
    all_timestamps : array-like, 1d
        All timestamps in the ttl_dir.
    """
    channel_states = np.load(os.path.join(ttl_dir, "channel_states.npy"))
    all_timestamps = np.load(os.path.join(ttl_dir, "timestamps.npy"))
    target_timestamps = all_timestamps[channel_states == target_channel]
    return target_timestamps, all_timestamps


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
    timestamps_aligned = np.round(timestamps_aligned).astype(int)
    return timestamps_aligned


def align_and_save_timestamps(
    to_be_aligned_sync_ch, ref_sync_ch, to_be_aligned_ttl_dir, ref_ttl_dir
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

    RETURN
    ------
    new_ttl_dir : str
        The path to the folder containing aligned TTLs/timestamps for TTLs
        in to_be_aligned_ttl_dir.
    """
    timestamps_sync, timestamps = get_timestamps(
        to_be_aligned_ttl_dir, to_be_aligned_sync_ch
    )
    reference_sync, timestamps_ref = get_timestamps(ref_ttl_dir, ref_sync_ch)
    new_ttl_dir = to_be_aligned_ttl_dir + "_aligned"
    if not os.path.exists(new_ttl_dir):
        os.makedirs(new_ttl_dir)
    timestamps_aligned = align_timestamps(timestamps, timestamps_sync, reference_sync)
    np.save(os.path.join(new_ttl_dir, "timestamps.npy"), timestamps_aligned)
    shutil.copy(os.path.join(to_be_aligned_ttl_dir, "channel_states.npy"), new_ttl_dir)
    return new_ttl_dir
