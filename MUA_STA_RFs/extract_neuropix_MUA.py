#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:18:27 2019

@author: kailun
"""

import numpy as np
from joblib import Parallel, delayed
from time import time
import os
from . import util


class neuropixData:
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
        raw_data_fname=None,
        fname_extensions=(None, None),
        total_ch=384,
        event_keys=[(1, "sync"), (2, "starts"), (3, "frametimes"), (4, "stops")],
        align_to_probe_timestamps=True,
        stim_sync_ch=1,
        probe_sync_ch=1,
        probe_ttl_dir="",
        chStep=1, 
        toBeAlignedTimestampsInfo=None,
        isSpikeGlx=False, 
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
        raw_data_fname : str or None
            The filename of the raw data. If None, 'continuous.dat' will be used.
        fname_extensions : tuple, optional
            The labels (prefix and postfix) for the files to be saved.
        total_ch : int
            The total number of Neuropixels channels in use.
        event_keys : list
            List containing tuples of channel states (int) and their corresponding
            TTL keys (str: 'starts', 'frametimes', 'stops', 'camera', etc.).
            E.g. [(1, 'starts'), (2, 'stim_frametimes'), ...].
        align_to_probe_timestamps : bool
            If True, the stimulus TTLs/timestamps will be aligned to the probe's
            clock (via probe sync TTLs) to compensate for the offsets in sampling rate.
        stim_sync_ch : int
            The channel state for the sync channel of the stimulus TTLs.
        probe_sync_ch : int
            The channel state for the sync channel of the probe TTLs.
        probe_ttl_dir : str
            The folder path of the probe TTLs.
        chStep : int
            The step in taking the channels to be extracted.
        toBeAlignedTimestampsInfo : list or None
            A list containing tuples of additional timestamps (besides the TTLs)
            file info to be aligned. The timestamps are assumed to be at the 
            same clock as the stimulus TTLs and will be aligned to the probe's
            clock time.
            E.g., [(key1, timestampsPath1, statesOrInfoPath1), 
                   (key2, timestampsPath2, statesOrInfoPath2), ...]
        isSpikeGlx : bool
            If True, the total data channels will be total_ch - 1, because 
            the last channel is the sync channel.
        """
        self.stim_ttl_dir = stim_ttl_dir
        self.raw_data_dir = raw_data_dir
        self.save_dir = save_dir
        self.raw_data_fname = raw_data_fname
        self._get_output_filename(fname_extensions)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.total_ch = total_ch
        self.event_keys = event_keys
        self.align_to_probe_timestamps = align_to_probe_timestamps
        self.stim_sync_ch = stim_sync_ch
        self.probe_sync_ch = probe_sync_ch
        self.probe_ttl_dir = probe_ttl_dir
        self.chStep = chStep
        self.toBeAlignedTimestampsInfo = toBeAlignedTimestampsInfo
        self.isSpikeGlx = isSpikeGlx
        self._alignAndExtractTimestamps()
        
    def extractMua(
            self, spike_event_std_thresh=-4, extractStartTimeSec=None,
            extractStopTimeSec=None, sliceLenSec=None, periStimExtractionKeys=None,
            periStimPreDurationSecs=[10], periStimTotDurationSecs=[20], 
            n_cores=int(os.cpu_count()/2)):
        """To extract spike times (MUA).
        PARAMETERS
        ----------
        spike_event_std_thresh : int or float
            The multiple of standard deviation of Butterworth bandpass filtered
            signals to be considered as spiking events.
        extractStartTimeSec : int or None
            The start time (in second, Neuropixels time) of the data to be extracted.
            If None, the data will be extracted from the beginning.
        extractStopTimeSec : int or None
            The end time (in second, Neuropixels time) of the data to be extracted.
            If None, the data will be extracted until the end.
        sliceLenSec : int
            The length (in second) for slicing the data, in case the data size is too big.
        periStimExtractionKeys : list or None
            The event/TTL keys for extracting the peri-stimulus MUA 
            (instead of extracting everything). Any overlapping slices will be
            merged. Only the data within the extract start and stop times will
            be extracted.
        periStimPreDurationSecs : list of int or float
            The durations (correspond to periStimExtractionKeys) in second 
            before the stimulus onsets/TTLs for the peri-stimulus extraction.
        periStimTotDurationSecs : list of int or float
            The total durations (correspond to periStimExtractionKeys) 
            in second for the peri-stimulus extraction.
        n_cores : int
            The number of CPUs to be used for parallel computing.
        """
        self.spike_event_std_thresh = spike_event_std_thresh
        self.extractStartTimeSec = extractStartTimeSec
        self.extractStopTimeSec = extractStopTimeSec
        self.sliceLenSec = sliceLenSec
        self.periStimExtractionKeys = periStimExtractionKeys
        self.periStimPreDurationSecs = periStimPreDurationSecs
        self.periStimTotDurationSecs = periStimTotDurationSecs
        self._n_cores = max(1, n_cores)
        if self.extractStartTimeSec is not None:
            self.extractStartTimeUt = int(round(
                self.extractStartTimeSec * self._sampling_rate))
        if self.extractStopTimeSec is not None:
            self.extractStopTimeUt = int(round(
                self.extractStopTimeSec * self._sampling_rate))
        
        if not os.path.exists(os.path.join(self.save_dir, self.spiketimes_fname)):
            self._start_extract_spikes()
            print("All spiketimes saved!")
        else:
            print("Spiketimes exists!")
        
    def _alignAndExtractTimestamps(self):
        if self.raw_data_fname is None:
            self.raw_data_fname = "continuous.dat"
        self.rawDataPath = os.path.join(self.raw_data_dir, self.raw_data_fname)
        self.totalDataCh = self.total_ch
        if self.isSpikeGlx:
            self.totalDataCh -= 1   # the last channel is sync channel
            util.genDataTimestamps(self.rawDataPath, self.total_ch)
            util.saveCsvToNeuropixTimestampsFormat(
                self.stim_ttl_dir, prefix='ttl_', samplingRate=self._sampling_rate)
            if self.probe_ttl_dir:
                util.saveCsvToNeuropixTimestampsFormat(
                    self.probe_ttl_dir, prefix='ttl_', 
                    samplingRate=self._sampling_rate)
        self.totalDataCh = len(np.arange(0, self.totalDataCh, self.chStep))
        self._get_data_path_info()
        if self.align_to_probe_timestamps:
            # align additional timestamps first before changing the stim_ttl_dir
            self._alignExtraTimestamps()
            
            # stim_ttl_dir is changed to a new stim_ttl_dir that contains the aligned TTLs
            self.stim_ttl_dir = util.align_and_save_timestamps(
                self.stim_sync_ch,
                self.probe_sync_ch,
                self.stim_ttl_dir,
                self.probe_ttl_dir,
                self.chState_fname,
                self.unitTimestamps_fname
            )
            self.stim_chState_fpath = os.path.join(
                self.stim_ttl_dir, self.chState_fname)
            self.stim_unitTimestamps_fpath = os.path.join(
                self.stim_ttl_dir, self.unitTimestamps_fname)
            print("The aligned stimulus TTL folder is {}.".format(self.stim_ttl_dir))
        self._extractTtls()
        
    def _get_output_filename(self, fname_extensions=None):
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
        
    def _get_data_path_info(self):
        latest_to_oldest_chStates_fnames = ["states.npy", "channel_states.npy"]
        latest_to_oldest_unitTimestamps_fnames = ["sample_numbers.npy", "timestamps.npy"]
        self.stim_chState_fpath = util.getLatestFilePath(
            self.stim_ttl_dir, latest_to_oldest_chStates_fnames)
        self.stim_unitTimestamps_fpath = util.getLatestFilePath(
            self.stim_ttl_dir, latest_to_oldest_unitTimestamps_fnames)
        _, self.chState_fname = os.path.split(self.stim_chState_fpath)
        _, self.unitTimestamps_fname = os.path.split(self.stim_unitTimestamps_fpath)
        
    def _alignExtraTimestamps(self):
        """The timestamps to be aligned are assumed to be at the same clock as 
        the stimulus TTLs and will be aligned to the probe's clock time.
        """
        if self.toBeAlignedTimestampsInfo is not None:
            assert isinstance(self.toBeAlignedTimestampsInfo, list), \
                'The toBeAlignedTimestampsInfo should be a list of tuples.'
            assert len(self.probe_ttl_dir) > 0, 'No probe_ttl_dir is specified!'
            
            stim_chState_fpath = os.path.join(
                self.stim_ttl_dir, self.chState_fname)
            stim_timestamps_fpath = os.path.join(
                self.stim_ttl_dir, self.unitTimestamps_fname)
            ref_chState_fpath = os.path.join(
                self.probe_ttl_dir, self.chState_fname)
            ref_timestamps_fpath = os.path.join(
                self.probe_ttl_dir, self.unitTimestamps_fname)
            timeSync, _ = util.get_timestamps(
                stim_timestamps_fpath, stim_chState_fpath, self.stim_sync_ch)
            refSync, _ = util.get_timestamps(
                ref_timestamps_fpath, ref_chState_fpath, self.probe_sync_ch)
            
            self.stim_ttl_dict = {}
            for timeInfo in self.toBeAlignedTimestampsInfo:
                key, timesPath, statesOrInfoPath = timeInfo
                times = np.load(timesPath, allow_pickle=True, encoding='latin1')
                info = np.load(statesOrInfoPath, allow_pickle=True, encoding='latin1')
                alignedTimes = util.align_timestamps(times, timeSync, refSync)
                self.stim_ttl_dict[key] = {}
                self.stim_ttl_dict[key]['alignedTimes'] = alignedTimes
                self.stim_ttl_dict[key]['info'] = info
        
    def _extractTtls(self):
        if not os.path.exists(os.path.join(self.save_dir, self.stim_ttl_fname)):
            self._get_stim_ttl()
            self.save(self.stim_ttl_dict, self.stim_ttl_fname)
            print("Done extracting stimulus TTLs!")
        else:
            print("Stimulus TTLs exists!")

    def _get_stim_ttl(self):
        """To get the stimulus TTLs/timestamps."""
        if not 'stim_ttl_dict' in self.__dict__:
            self.stim_ttl_dict = {}
        channel_states = np.load(self.stim_chState_fpath)
        timestamps = np.load(self.stim_unitTimestamps_fpath)
        for ch, key in self.event_keys:
            ch_timestamps = timestamps[channel_states==ch]
            self.stim_ttl_dict[key] = ch_timestamps

    def _start_extract_spikes(self):
        """Start extract the spike times slice-by-slice."""
        self._get_raw_data()
        self._flatten_timestamps()
        self._getSliceInd()
        self._curSliceIsEmpty = False
        for n, start in enumerate(self._sliceStartInd, 1):
            if not os.path.exists(os.path.join(self.save_dir, "slice{}.npy".format(n))):
                t1 = time()
                self._cur_start = start
                self._cur_stop = self._sliceStopInd[n-1]
                print(
                    "Current slice (unit time): {} to {}".format(
                        self._cur_start, self._cur_stop
                    )
                )
                self._get_cur_data()
                if self._curSliceIsEmpty:
                    print('No more spike data to extract!')
                    n -= 1
                    break
                self._get_spiketimes(n)
                print("Slice {} done! Time: {:.2f}s".format(n, time() - t1))
            else:
                print("Slice {} exists!".format(n))
        self._save_spiketimes_dict(n)
        del self.raw_data
        if os.path.exists(os.path.join(self.save_dir, "temp.npy")):
            if not self._curSliceIsEmpty:
                del self.cur_data
            os.remove(os.path.join(self.save_dir, "temp.npy"))

    def _get_raw_data(self):
        """To get the raw_data_timestamps and raw_data."""
        self.raw_data_timestamps = np.load(
            os.path.join(self.raw_data_dir, self.unitTimestamps_fname)
        )
        self.raw_data = np.memmap(self.rawDataPath, dtype=np.int16, mode="r")
        self.raw_data = self.raw_data.reshape((self.total_ch, -1), order="F")
        self.raw_data = self.raw_data[::self.chStep]
        self.rawDataLen = self.raw_data.shape[1]

    def _flatten_timestamps(self):
        """To correct the raw_data_timestamps (if any mistakes) and get the 
        data_start_time and total_timestamp_len.
        """
        (self.data_start_time, self.total_timestamp_len
         ) = util.get_rawdata_timestamps_info(self.raw_data_timestamps)
        start_zero = self.data_start_time == 0
        correct_ideal_len = self.rawDataLen - self.total_timestamp_len == 0
        correct_time_len = (
            self.total_timestamp_len - self.raw_data_timestamps.shape[0] == 0
        )
        if start_zero:
            self.raw_data_timestamps = np.arange(self.rawDataLen)
        else:
            if not correct_ideal_len:
                if (
                    self.data_start_time + self.total_timestamp_len
                    == self.rawDataLen
                ):
                    self.raw_data_timestamps = np.arange(self.rawDataLen)
                else:
                    util.get_action(
                        "First timestamp (data_start_time): {}\n"
                        "The raw data ({}) and timestamps ({}) lengths"
                        " are not consistent!".format(
                            self.data_start_time,
                            self.rawDataLen,
                            self.total_timestamp_len,
                        )
                    )
            else:
                if not correct_time_len:
                    self.raw_data_timestamps = np.arange(
                        self.data_start_time,
                        self.data_start_time + self.rawDataLen,
                    )
    
    def _getSliceInd(self):
        self.extractStartIdx = (
            0
            if not self.extractStartTimeSec
            else self.extractStartTimeUt - self.data_start_time
        )
        self.extractStopIdx = (
            self.rawDataLen
            if not self.extractStopTimeSec
            else self.extractStopTimeUt - self.data_start_time
        )
        self.extractStartIdx = np.clip(
            self.extractStartIdx, a_min=0, a_max=self.rawDataLen)
        self.extractStopIdx = np.clip(
            self.extractStopIdx, a_min=0, a_max=self.rawDataLen)
        if self.sliceLenSec:
            self.sliceLenUt = self.sliceLenSec * self._sampling_rate
            ind = np.arange(self.extractStartIdx, self.extractStopIdx, self.sliceLenUt)
            sliceInd = np.append(ind, self.extractStopIdx)
        else:
            self.sliceLenUt = None
            sliceInd = np.array([self.extractStartIdx, self.extractStopIdx])
        self._sliceStartInd = sliceInd[:-1]
        self._sliceStopInd = sliceInd[1:]
        
        if self.periStimExtractionKeys is not None:
            self._sliceStartInd = []
            self._sliceStopInd = []
            self._addPeriStimSliceInd()
        slices = list(zip(self._sliceStartInd, self._sliceStopInd))
        print(
            f"Slices (unit time): {slices}\nTotal number of slices: {len(slices)}")
    
    def _addPeriStimSliceInd(self):
        """Add the peri-stimulus slices to the slices to be extracted. If there
        are overlaps between the existing slices and peri-stimulus slices, the
        fully/partially overlapped peri-stimulus slices will be merged.
        """
        stimTtlPath = os.path.join(self.save_dir, self.stim_ttl_fname)
        stimTtlDict = np.load(stimTtlPath, allow_pickle=True, encoding='latin1').item()
        for i, key in enumerate(self.periStimExtractionKeys):
            periStimTtls = stimTtlDict[key]
            periStimPreDurationUt = int(round(self.periStimPreDurationSecs[i] * self._sampling_rate))
            periStimTotDurationUt = int(round(self.periStimTotDurationSecs[i] * self._sampling_rate))
            periStimStartInd = periStimTtls - periStimPreDurationUt
            periStimStartInd -= self.data_start_time
            periStimStopInd = periStimStartInd + periStimTotDurationUt
            self._sliceStartInd = np.append(self._sliceStartInd, periStimStartInd).astype(int)
            self._sliceStopInd = np.append(self._sliceStopInd, periStimStopInd).astype(int)
        self._sliceStartInd = np.clip(
            self._sliceStartInd, a_min=self.extractStartIdx, a_max=self.extractStopIdx)
        self._sliceStopInd = np.clip(
            self._sliceStopInd, a_min=self.extractStartIdx, a_max=self.extractStopIdx)
        self._sliceStartInd, self._sliceStopInd = util.mergeOverlappedTrials(
            self._sliceStartInd, self._sliceStopInd, self.sliceLenUt)
    
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
        try:
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
        except ValueError:
            self._curSliceIsEmpty = True
            print('Current slice is empty!')

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
            delayed(util.filter_detect)(
                self.cur_data[ch - 1, :],
                self.raw_data_timestamps[self._cur_start],
                self.spike_event_std_thresh,
                self._signal_orientation,
                self._signal_lowcut,
                self._signal_highcut,
                self._sampling_rate,
                self._butter_bandpass_order,
            )
            for ch in range(1, self.totalDataCh + 1)
        )
        spiketimesDict = {i:arr for i, arr in enumerate(spiketimes_arr)}
        self.save(spiketimesDict, "slice{}".format(slice_idx))

    def _save_spiketimes_dict(self, num_files):
        """Load saved spiketimes file of slices, put them in a dict, and save dict.
        The spiketimes are not put in dict and saved because the dict will slow down the performance of multiple cores.
        PARAMETERS
        ----------
        num_files : int
            The total number of files to be loaded and saved.
        """
        self.st_dict = {ch: [] for ch in range(1, self.totalDataCh + 1)}
        for i in range(num_files):
            cur_file = os.path.join(self.save_dir, "slice{}.npy".format(i + 1))
            st_arr = np.load(cur_file, allow_pickle=True).item()
            for ch in range(1, self.totalDataCh + 1):
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
