#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:18:27 2019

@author: kailun
"""

import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
from time import time
import os
from .util import *


class extract_NP_MUA:
    _signal_lowcut = 300  # the low cut of the signal frequency
    _signal_highcut = 3000  # the high cut of the signal frequency
    _butter_bandpass_order = 2  # the order for the Butterworth bandpass filter
    _sampling_rate = 30000  # the sampling rate of the Neuropixels probe
    _signal_orientation = (
        "negative"  # 'positive', 'negative', or 'both', the signal orientation for processing
    )

    def __init__(
        self,
        stim_ttl_dir,
        raw_data_dir,
        save_dir,
        fname_extensions=(None, None),
        total_ch=384,
        spike_event_std_thresh=-4,
        extract_start_time=None,
        extract_stop_time=None,
        event_keys=[(1, "sync"), (2, "starts"), (3, "frametimes"), (4, "stops")],
        slice_len=None,
        align_to_probe_timestamps=True,
        stim_sync_ch=1,
        probe_sync_ch=1,
        probe_ttl_dir="",
        chState_filename="channel_states.npy",
        n_cores=mp.cpu_count(),
    ):
        """To extract the TTLs/timestamps and MUA from raw Neuropixels data.
        PARAMETERS
        ----------
        stim_ttl_dir : str
            The folder path of the stimulus TTLs (path until .../TTL_1).
        raw_data_dir : str
            The folder path of raw data (path until .../Neuropix-3a-100.0).
        save_dir : str
            The folder path for saving the outputs.
        fname_extensions : tuple, optional
            The labels (prefix and postfix) for the files to be saved.
        total_ch : int
            The total number of Neuropixels channels in use.
        spike_event_std_thresh : int or float
            The multiple of standard deviation of Butterworth bandpass filtered
            signals to be considered as spiking events.
        extract_start_time : int, optional
            The start time (in unit time) of the data to be extracted.
            If None, the data will be extracted from the beginning.
        extract_stop_time : int, optional
            The end time (in unit time) of the data to be extracted.
            If None, the data will be extracted until the end.
        event_keys : list
            List containing tuples of channel states (int) and their corresponding
            TTL keys (str: 'starts', 'frametimes', 'stops', 'camera', etc.).
            E.g. [(1, 'starts'), (2, 'stim_frametimes'), ...].
        slice_len : int
            The length (in second) for slicing the data, in case the data size is too big.
        align_to_probe_timestamps : bool
            If True, the stimulus TTLs/timestamps will be aligned to the probe's
            clock (via probe sync TTLs) to compensate for the offsets in sampling rate.
        stim_sync_ch : int
            The channel state for the sync channel of the stimulus TTLs.
        probe_sync_ch : int
            The channel state for the sync channel of the probe TTLs.
        probe_ttl_dir : str
            The folder path of the probe TTLs.
        n_cores : int
            The number of CPUs to be used for parallel computing.
        """
        self.stim_ttl_dir = stim_ttl_dir
        self.raw_data_dir = raw_data_dir
        self.save_dir = save_dir
        self._get_filename(fname_extensions)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.total_ch = total_ch
        self.spike_event_std_thresh = spike_event_std_thresh
        self.extract_start_time = extract_start_time
        self.extract_stop_time = extract_stop_time
        self.event_keys = event_keys
        self.slice_len = slice_len
        self.align_to_probe_timestamps = align_to_probe_timestamps
        self.stim_sync_ch = stim_sync_ch
        self.probe_sync_ch = probe_sync_ch
        self.probe_ttl_dir = probe_ttl_dir
        self.chState_filename = chState_filename
        if self.align_to_probe_timestamps:
            # stim_ttl_dir is changed to a new stim_ttl_dir that contains the aligned TTLs
            self.stim_ttl_dir = align_and_save_timestamps(
                self.stim_sync_ch,
                self.probe_sync_ch,
                self.stim_ttl_dir,
                self.probe_ttl_dir,
                self.chState_filename,
            )
            print("The aligned stimulus TTL folder is {}.".format(self.stim_ttl_dir))
        self._n_cores = max(1, n_cores)
        self._extract()

    def _get_filename(self, fname_extensions=None):
        """To get the filename for stimulus TTLs and MUA spiketimes."""
        if fname_extensions:
            prefix, postfix = fname_extensions
        else:
            prefix, postfix = None, None
        self.stim_ttl_fname = (
            "stim_TTLs.npy" if not prefix else prefix + "stim_TTLs.npy"
        )
        st_fname = "spiketimes" if not prefix else prefix + "spiketimes"
        self.spiketimes_fname = st_fname if not postfix else st_fname + postfix
        self.spiketimes_fname += ".npy"

    def _extract(self):
        """To extract the stimulus TTLs/timestamps and spike times (MUA)."""
        self.stim_timestamps, self.raw_data_timestamps = standardize_timestamps(
                [self.stim_ttl_dir, self.raw_data_dir], self._sampling_rate)
        if not os.path.exists(os.path.join(self.save_dir, self.stim_ttl_fname)):
            self._get_stim_ttl()
            self.save(self.stim_ttl_dict, self.stim_ttl_fname)
            print("Done extracting stimulus TTLs!")
        else:
            print("Stimulus TTLs exists!")
        if not os.path.exists(os.path.join(self.save_dir, self.spiketimes_fname)):
            self._start_extract_spikes()
            print("All spiketimes saved!")
        else:
            print("Spiketimes exists!")

    def _get_stim_ttl(self):
        """To get the stimulus TTLs/timestamps."""
        self.stim_ttl_dict = {}
        channel_states = np.load(os.path.join(self.stim_ttl_dir, self.chState_filename))
        for ch, key in self.event_keys:
            ch_timestamps = self.stim_timestamps[channel_states == ch]
            start_t = self.extract_start_time if self.extract_start_time else 0
            if self.extract_stop_time:
                self.stim_ttl_dict[key] = ch_timestamps[
                    np.where(
                        (ch_timestamps >= start_t)
                        & (ch_timestamps <= self.extract_stop_time)
                    )
                ]
            else:
                self.stim_ttl_dict[key] = ch_timestamps[
                    np.where(ch_timestamps >= start_t)
                ]

    def _start_extract_spikes(self):
        """Start extract the spike times slice-by-slice."""
        self._get_raw_data()
        self._flatten_timestamps()
        extract_start_idx = (
            0
            if not self.extract_start_time
            else self.extract_start_time - self.data_start_time
        )
        extract_stop_idx = (
            self.raw_data.shape[1]
            if not self.extract_stop_time
            else self.extract_stop_time - self.data_start_time
        )
        if self.slice_len:
            idx = np.arange(
                extract_start_idx, extract_stop_idx, self.slice_len * self._sampling_rate
            )
            idx = np.append(idx, extract_stop_idx)
        else:
            idx = np.array([extract_start_idx, extract_stop_idx])
        print(
            "Slicing points (unit time): {}\nTotal number of slices: {}".format(
                idx, len(idx) - 1
            )
        )
        for n, i in enumerate(idx[:-1], 1):
            if not os.path.exists(os.path.join(self.save_dir, "slice{}.npy".format(n))):
                t1 = time()
                self._cur_start = i
                self._cur_stop = idx[n]
                print(
                    "Current slice (unit time): {} to {}".format(
                        self._cur_start, self._cur_stop
                    )
                )
                self._get_cur_data()
                self._get_spiketimes(n)
                print("Slice {} done! Time: {:.2f}s".format(n, time() - t1))
            else:
                print("Slice {} exists!".format(n))
        self._save_spiketimes_dict(n)
        del self.raw_data
        if os.path.exists(os.path.join(self.save_dir, "temp.npy")):
            del self.cur_data
            os.remove(os.path.join(self.save_dir, "temp.npy"))

    def _get_raw_data(self):
        """To get the raw_data_timestamps and raw_data."""
        cur_data_path = os.path.join(self.raw_data_dir, "continuous.dat")
        self.raw_data = np.memmap(cur_data_path, dtype=np.int16, mode="r")
        self.raw_data = self.raw_data.reshape((self.total_ch, -1), order="F")

    def _flatten_timestamps(self):
        """To correct the raw_data_timestamps (if any mistakes) and get the data_start_time and total_timestamp_len."""
        self.data_start_time, self.total_timestamp_len = get_rawdata_timestamps_info(
            self.raw_data_timestamps
        )
        start_zero = self.data_start_time == 0
        correct_ideal_len = self.raw_data.shape[1] - self.total_timestamp_len == 0
        correct_time_len = (
            self.total_timestamp_len - self.raw_data_timestamps.shape[0] == 0
        )
        if start_zero:
            self.raw_data_timestamps = np.arange(self.raw_data.shape[1])
        else:
            if not correct_ideal_len:
                if (
                    self.data_start_time + self.total_timestamp_len
                    == self.raw_data.shape[1]
                ):
                    self.raw_data_timestamps = np.arange(self.raw_data.shape[1])
                else:
                    get_action(
                        "First timestamp (data_start_time): {}\n"
                        "The raw data ({}) and timestamps ({}) lengths"
                        " are not consistent!".format(
                            self.data_start_time,
                            self.raw_data.shape[1],
                            self.total_timestamp_len,
                        )
                    )
            else:
                if not correct_time_len:
                    self.raw_data_timestamps = np.arange(
                        self.data_start_time,
                        self.data_start_time + self.raw_data.shape[1],
                    )

    def _get_cur_data(self):
        """To get the data of current slice."""
        t0 = time()
        tmp_file = os.path.join(self.save_dir, "temp.npy")
        if os.path.exists(tmp_file):
            try:
                del self.cur_data
                os.remove(tmp_file)
            except AttributeError:
                pass
        self.cur_data = np.memmap(
            tmp_file,
            dtype=np.int16,
            mode="w+",
            shape=self.raw_data[:, self._cur_start : self._cur_stop].shape,
        )
        t1 = time()
        print("Open file time: {:.2f}s".format(t1 - t0))
        self.cur_data[:] = self.raw_data[:, self._cur_start : self._cur_stop]
        t2 = time()
        print("Copy raw data slice time: {:.2f}s".format(t2 - t1))

        # subtract the common average reference
        self._median_subtraction(0)
        t3 = time()
        print("Done subtracting axis-0 median: {:.2f}s".format(t3 - t2))
        # offset correction for Phase3A probes
        self._median_subtraction(1)
        t4 = time()
        print("Done subtracting axis-1 median: {:.2f}s".format(t4 - t3))

    def _get_spiketimes(self, slice_idx):
        """To detect and extract the spiketimes (MUA).
        PARAMETERS
        ----------
        slice_idx : int
            The index of the current data slice.

        NOTE
        ----
        The order of variables in raw data file:
            sampling frequency (adfrequency),
            length of the recording (n),
            timestamps,
            last recording time stamps (fragmentcounts),
            raw data (ad).
        spiketimes_arr gives the spiketimes in unit time for each channel.
        """
        spiketimes_arr = Parallel(n_jobs=self._n_cores, verbose=10)(
            delayed(filter_detect)(
                self.cur_data[ch - 1, :],
                self.raw_data_timestamps[self._cur_start],
                self.spike_event_std_thresh,
                self._signal_orientation,
                self._signal_lowcut,
                self._signal_highcut,
                self._sampling_rate,
                self._butter_bandpass_order,
            )
            for ch in range(1, self.total_ch + 1)
        )
        self.save(spiketimes_arr, "slice{}".format(slice_idx))

    def _save_spiketimes_dict(self, num_files):
        """Load saved spiketimes file of slices, put them in a dict, and save dict.
        The spiketimes are not put in dict and saved because the dict will slow down the performance of multiple cores.
        PARAMETERS
        ----------
        num_files : int
            The total number of files to be loaded and saved.
        """
        self.st_dict = {ch: [] for ch in range(1, self.total_ch + 1)}
        for i in range(num_files):
            cur_file = os.path.join(self.save_dir, "slice{}.npy".format(i + 1))
            st_arr = np.load(cur_file, allow_pickle=True)
            for ch in range(1, self.total_ch + 1):
                self.st_dict[ch] = np.append(self.st_dict[ch], st_arr[ch - 1]).astype(
                    int
                )
            os.remove(cur_file)
        self.save(self.st_dict, self.spiketimes_fname)

    def _median_subtraction(self, axis):
        """To subtract the median from the cur_data.
        PARAMETERS
        ----------
        axis : int
            The axis of the cur_data to be used for median subtraction.
        """
        length = self.cur_data.shape[1] if axis == 0 else self.cur_data.shape[0]
        idx = np.linspace(0, length, self._n_cores + 1).astype(int)
        Parallel(n_jobs=self._n_cores)(
            delayed(self._parallel_median)(idx[i], idx[i + 1], axis)
            for i in range(self._n_cores)
        )

    def _parallel_median(self, start_idx, end_idx, axis):
        """Median subtraction subprocess for parallelization."""
        if axis == 0:
            median = np.median(self.cur_data[:, start_idx:end_idx], axis=0).astype(int)
            self.cur_data[:, start_idx:end_idx] -= median[np.newaxis, :]
        elif axis == 1:
            median = np.median(self.cur_data[start_idx:end_idx], axis=1).astype(int)
            self.cur_data[start_idx:end_idx] -= median[:, np.newaxis]

    def save(self, data, filename):
        """To save a data array to the save_dir.
        PARAMETERS
        ----------
        data : dict or array-like
            The data to be saved.
        filename : str
            The file name for the data to be saved.
        """
        np.save(os.path.join(self.save_dir, filename), data)
