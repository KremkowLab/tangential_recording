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
import os, shutil, time


def get_action(message):
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
    timestamps_len = timestamps.shape[0]
    start_time = timestamps[0]
    time_diff = np.diff(timestamps)
    (loss_time_idx,) = np.where(time_diff != 1)
    loss_time_len = time_diff[loss_time_idx].sum() - loss_time_idx.size
    total_timestamp_len = timestamps_len + loss_time_len
    return start_time, total_timestamp_len


def butter_bandpass(low_cut, high_cut, sampling_rate, order=2):
    """
    Butterworth bandpass filter from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    nyq = 0.5 * sampling_rate
    low = low_cut / nyq
    high = high_cut / nyq
    return butter(order, [low, high], btype="band")


def butter_bandpass_filter(data, low_cut, high_cut, sampling_rate, order=2):
    b, a = butter_bandpass(low_cut, high_cut, sampling_rate, order)
    y = filtfilt(b, a, data)
    return y


def basic_peak_detector(sig, orientation="negative", thresh=-3.5, verbose=False):
    """To detect spiking events."""
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
    return ind + 1


def filter_detect(
    channel,
    start,
    thresh,
    orientation,
    low_cut,
    high_cut,
    sampling_rate,
    order=2,
    verbose=False,
):
    data_filt = butter_bandpass_filter(channel, low_cut, high_cut, sampling_rate, order)
    threshold = thresh * np.std(data_filt)
    spiketimes_tmp = basic_peak_detector(data_filt, orientation, threshold, verbose)
    spiketimes_tmp += start
    return spiketimes_tmp


def get_timestamps(ttl_dir, target_channel):
    channel_states = np.load(os.path.join(ttl_dir, "channel_states.npy"))
    all_timestamps = np.load(os.path.join(ttl_dir, "timestamps.npy"))
    target_timestamps = all_timestamps[channel_states == target_channel]
    return target_timestamps, all_timestamps


def align_timestamps(timestamps, timestamps_sync, reference_sync, verbose=False):
    """
    To align NIDAQ timestamps to NP timestamps.

    timestamps: timestamps to be aligned (NIDAQ TTL timestamps)
    timestamp_sync: sync timestamps to be aligned (NIDAQ sync)
    reference_sync: sync timestamps as reference (NP sync)
    """
    # assert len(timestamps_sync) == len(reference_sync), "The timestamps_sync ({}) and reference_sync ({}) do not have same length.".format(len(timestamps_sync), len(reference_sync))
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
    return np.round(timestamps_aligned).astype(int)


def align_and_save_timestamps(
    to_be_aligned_sync_ch, ref_sync_ch, to_be_aligned_ttl_dir, ref_ttl_dir
):
    """
    PARAMETERS
    ----------
    to_be_aligned_sync_ch : int
        The channel state of the sync TTLs to be aligned.
    ref_sync_ch : int
        The channel state of the reference sync TTLs.
    to_be_aligned_ttl_dir: The NIDAQ TTL folder
    ref_ttl_dir: The Neuropixel TTL folder

    Note: the sync_channel is assumed to be the same for analog signals if given analog_ttl_dir.
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
