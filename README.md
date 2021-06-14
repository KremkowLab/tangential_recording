# Probe placement optimization for Neuropixels tangential recording

This repo implements the receptive field (RF) analysis via spike-triggered average (STA) of the multi-unit activity (MUA) for optimizing the placement of Neuropixels probe during tangential recordings in the mouse superior colliculus (SC).

Run the `MUA_STA_RFs_exec.py` to:
1. extract the MUA from the Neuropixels recording,
2. compute the STA of the MUA given the locally-sparse-noise stimuli for estimating the RF for each channel of the probe, and 
3. plot the estimated RFs for estimating and monitoring the relative probe location in the mouse SC.
