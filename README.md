# Probe placement optimization for Neuropixels tangential recordings

This repo implements the receptive field (RF) analysis via spike-triggered average (STA) of the multi-unit activity (MUA) from Neuropixels probe recordings. These scripts intend to quickly test the presence of any neuronal responses to a visual stimulus during tangential recordings in the mouse superior colliculus ([Sibille et al., 2021](https://www.biorxiv.org/content/10.1101/2021.06.12.448191v1.abstract)). We present a sparse noise stimulus to map the RFs. Each frame within the stimulus matrix is paired with a timestamp (TTL) to ensure proper alignment of the recorded neuronal activity to the timing of the visual stimulus frames. This code will produce the peri-stimulus time histogram (PSTH) and the corresponding RFs.

Run the `MUA_STA_RFs_exec.py` to:
1. extract the MUA from the Neuropixels recording,
2. compute the STA of the MUA to map the RFs for each channel of the probe, and 
3. plot the estimated RFs.


Software: 
1. Python, PsychoPy
(we provide an example of a short Neuropixels recording (5 GB) to troubleshoot your system)

Hardware:
1. You should have PsychoPy toolbox installed to expose the visual stimulus. Make sure to have matching numbers of TTLs between the stimulus matrix and the TTLs.
2. This script goes with two hardware options: the simpler option 1 for the script in its current state: the Stim TTL should simply be plugged into the front digital inputs of your Neuropixels card. The option 2 is possible if your have an additional NI-DAQ inputs in your NI-PXIe where your TTL should be recorded into the digital inputs number 3 (or change the "event_keys "frametimes" to your digital inputs). When using such an hardware settings an extra synchronization step between both card will be is required which will be activated by the options ('align_to_probe_timestamps=True' + write in your probe TTLs directors in "probe_ttl_dir="). 
