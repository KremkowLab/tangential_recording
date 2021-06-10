#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 18:33:52 2021

@author: kailun
"""

import numpy as np
import os
import matplotlib.pyplot as plt


class get_STA:
    def __init__(
        self, stimulus, spiketimes, frametimes, lags=np.arange(-5, 5), save_dir=""
    ):
        """To compute the STA for obtaining the RFs.
        PARAMETERS
        ----------
        stimulus : array-like, 3d
            Locally sparse noise stimuli with shape = (ny, nx, nframes).
        spiketimes : dict
            Dictionary containing the spiketimes for each channel.
            The dictionary keys are the channels.
        frametimes : array_like, 1d
            The stimulus timestamps/TTLs for the locally sparse noise frames.
        lags : list or array-like
            The lags for computing the STA.
        save_dir : str
            The folder path for saving the outputs.
        """
        self.stimulus = np.array(stimulus)
        self.spiketimes = spiketimes
        self.frametimes = frametimes
        self.lags = np.array(lags)
        self.save_dir = save_dir if save_dir else os.getcwd()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self._get_info()
        self._calc_STA()
        print(self)

    def __str__(self):
        """Class object info."""
        return (
            "\nTotal number of recording channels: {}\n"
            "Estimated frame duration (unit time): {:.5f} (std = {:.5f})\n"
            "Total number of stimulus frames: {}\n"
            "Stimulus dimensions (y, x): ({}, {})"
        ).format(
            self.n_tot_chs,
            self.frame_duration_avg,
            self.frame_duration_std,
            self.n_frames,
            self.n_y,
            self.n_x,
        )

    def _get_info(self):
        """To get the relevant information."""
        self.frame_duration_avg = np.diff(self.frametimes).mean()
        self.frame_duration_std = np.diff(self.frametimes).std()
        self.n_frames = len(self.frametimes)
        if self.n_frames - self.stimulus.shape[2] > 0:
            self.n_frames = self.stimulus.shape[2]
        self.all_channels = list(self.spiketimes.keys())
        self.n_tot_chs = len(self.all_channels)
        self.n_lags = len(self.lags)
        self.n_y, self.n_x = self.stimulus.shape[0:2]
        self.first_frame = -self.lags[0]
        self.last_frame = self.n_frames - self.lags[-1]
        self.ft_bins = np.append(
            self.frametimes, self.frametimes[-1] + self.frame_duration_avg
        )

    def _calc_STA(self):
        """To compute and save the STA."""
        self.STA = np.zeros((self.n_tot_chs, self.n_y, self.n_x, self.n_lags))
        for ch in range(self.n_tot_chs):
            st = self.spiketimes[ch + 1]
            n_spikes, _ = np.histogram(st, self.ft_bins)
            for i in range(self.first_frame, self.last_frame):
                # weight surrounding frames by the number of spikes
                self.STA[ch] += n_spikes[i] * self.stimulus[:, :, i + self.lags]
            self.STA[ch] -= self.STA[ch].mean()
            self.STA[ch] /= np.abs(self.STA[ch]).max()
        save_path = os.path.join(self.save_dir, "STA_arr.npy")
        np.save(save_path, self.STA)
        print("\nThe STA is saved as {}".format(save_path))

    def plot(self, subplots_rc=(20, 20), fig_fname="STA_RFs.png", fig_size_pix=None):
        """To plot and save the STA RFs.
        PARAMETERS
        ----------
        subplots_rc : list or tuple
            The number of rows and columns to plot the STA RFs.
        fig_fname : str
            The filename for saving the plotted figure.
        fig_size_pix : list or tuple
            The width and height (in pixel) of the figure to be plotted.

        RETURN
        ------
        fig : matplotlib object
            The plotted figure.
        """
        if subplots_rc:
            nrows, ncols = subplots_rc
        else:
            ncols = int(np.sqrt(self.n_tot_chs))
            nrows = np.ceil(self.n_tot_chs / ncols).astype(int)
        fig, axes = plt.subplots(nrows, ncols)
        fig.subplots_adjust(
            wspace=0.1, hspace=0.2, top=0.99, bottom=0.01, left=0.002, right=0.998
        )
        mngr = plt.get_current_fig_manager()
        if fig_size_pix:
            fig_width, fig_height = fig_size_pix
        else:
            scaling = 55
            fig_width = ncols * scaling
            fig_height = nrows * scaling
        mngr.window.setGeometry(0, 0, fig_width, fig_height)
        for i in range(nrows * ncols):
            ax = axes.ravel()[i]
            if i < self.n_tot_chs:
                idx = np.argmax(np.abs(self.STA[i]))
                x, y, frame = np.unravel_index(idx, self.STA[i].shape)
                pch = ax.pcolormesh(self.STA[i, :, :, frame], cmap="magma")
                ax.set_aspect("equal", "box")
                ax.set_title("ch {}".format(self.all_channels[i]), fontsize=10, y=0.95)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                pch.set_clim(-1, 1.0)
            else:
                ax.set_visible(False)
        plt.tight_layout(pad=0.4, w_pad=0.0, h_pad=0.0)
        plt.pause(5.0)  # needed for plotting first, then savefig
        save_path = os.path.join(self.save_dir, fig_fname)
        plt.savefig(save_path, orientation="landscape", bbox_inches="tight", dpi=600)
        print("\nThe STA RFs plot is saved as {}".format(save_path))
        return fig
