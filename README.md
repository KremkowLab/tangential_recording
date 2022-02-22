# Probe placement optimization for Neuropixels tangential recording

This repo implements the receptive field (RF) analysis via spike-triggered average (STA) of the multi-unit activity (MUA) from Neuropixels probe recordings. This is made for optimizing the placement of probe in any region exhibiting visually driven activity, as for example to perform tangential recordings in the mouse superior colliculus (SC, ref???). You should proceed first having the sparse noise exposed to the animal while you record the activity from the Neuropixels probe and then use this script to have the RF plotted for each channels.

Run the `MUA_STA_RFs_exec.py` to:
1. extract the MUA from the Neuropixels recording,
2. compute the STA of the MUA given the locally-sparse-noise stimuli for estimating the RF for each channel of the probe, and 
3. plot the estimated RFs for estimating and monitoring the relative probe location in the mouse SC.

NB: you should have python 3 or above.
