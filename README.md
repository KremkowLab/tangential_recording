# Probe placement optimization for Neuropixels tangential recording

This repo implements the receptive field (RF) analysis via spike-triggered average (STA) of the multi-unit activity (MUA) from Neuropixels probe recordings. These scripts are made to quickly probe the presence of any neuronal responses to visual stimulus in any region covered by the probe during tangential recordings in the mouse superior colliculus ([Sibille et al., 2021](https://www.biorxiv.org/content/10.1101/2021.06.12.448191v1.abstract)). You should produce beforehand a sparse noise stimulus matrix and have it exposed in one of your screen within the set-up while producing an electrical pulse (TTL) to each exposed images into one of the digital inputs of your neuropixel recording system. This pulse is what we define as the "Stim TTL" and will guarantee a proper alignement of the recorded neuronal activity from the Neuropixels probe to the timing of the exposed stimulus which will produce the PSTH and then the STA-RF.

Run the `MUA_STA_RFs_exec.py` to:
1. extract the MUA from the Neuropixels recording,
2. compute the STA of the MUA given the locally-sparse-noise stimuli for estimating the RF for each channel of the probe, and 
3. plot the estimated RFs for estimating and monitoring the relative probe location in the tissue.


NB software: 
1. you should have python 3 or above.
2. you should probe all of your scripts before recording day, we can provide a "schoolbook example" of a short Neuropixel recording (5 Go) to troubleshoot your system.

NB hardware:
1. You should have pylon viewer (or any equivalent) to exposed the sparse noise in a reliable way. Make sure you exposed one image every 50 ms to spare time but also make sure you have matching number of TTLs between the stimulation matrice and the analyzed TTLs.
2. This script goes with two hardware options: the simpler options 1 for the script in its current state: the Stim TTL should simply be plugged into the front digital inputs of your Neuropixels card. The option 2 is possible if your have an additional NI-DAQ inputs in your NI-PXIe where your TTL should be recorded into the digital inputs number 3 (or change the "event_keys "frametimes" to your digital inputs). When using such an hardware settings an extra synchronization step between both card will be is required which will be activated by the options ('align_to_probe_timestamps=True' + write in your probe TTLs directors in "probe_ttl_dir="). 
