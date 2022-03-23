# Probe placement optimization for Neuropixels tangential recording

This repo implements the receptive field (RF) analysis via spike-triggered average (STA) of the multi-unit activity (MUA) from Neuropixels probe recordings. This scripts are made to quickly probe the presence of any neuronal responses to visual stimulus in any region covered by the probe, as adviced to perform tangential recordings in the mouse superior colliculus (Sibille Kremkow J. Neurosci. 2022). You should produce first a sparse noise matrice and have it exposed in one of your screen within the set-up while synchronizing an electrical pulse copied for each exposed images into one of the digital inputs of your neuropixel recording system. This pulse is what we define as the TTL and will guarantee a proper alignement of the recorded neuronal activity from the Neuropixels probe to the timing of the exposed stimulus which will produce the PSTH and then the STA-RF.

Run the `MUA_STA_RFs_exec.py` to:
1. extract the MUA from the Neuropixels recording,
2. compute the STA of the MUA given the locally-sparse-noise stimuli for estimating the RF for each channel of the probe, and 
3. plot the estimated RFs for estimating and monitoring the relative probe location in the mouse SC.


NB software: 
1- you should have python 3 or above. /%
2- you should probe all of your scripts before recording day, we can provide a "schoolbook example" of a short Neuropixel recording (5 Go) to troubleshoot your system.

NB hardware:
1- You should have pylon viewer (or any equivalent) to exposed the sparse noise in a reliable way: make sure you exposed one image every 50 ms to spare time./%
2- Your TTL should be recorded into an additional NI-DAQ plugged onto your Neuropixel recording system. When doing so an extra synchronization between both card will be is required before which is included in our repo. For this options a second synchronizing TTL from the Neuropixel card into the NI-DAQ card in the NI-PXIe will become the "syncrhonizing TTL".
