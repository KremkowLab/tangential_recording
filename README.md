# Probe placement optimization for Neuropixels tangential recordings

This repo implements the receptive field (RF) analysis via spike-triggered average (STA) of the multi-unit activity (MUA) from Neuropixels probe recordings. These scripts intend to quickly test the presence of any neuronal responses to visual stimuli in order to optimize the probe placement during tangential recordings in the mouse superior colliculus ([Sibille et al., 2021](https://www.biorxiv.org/content/10.1101/2021.06.12.448191v1.abstract)). Specifically, sparse-noise stimuli were presented to map the RFs. The peri-stimulus time histograms (PSTHs) and the corresponding RFs will be produced as part of the outputs.

Run the `MUA_STA_RFs_exec.py` to:
1. extract the MUA from the Neuropixels recording,
2. compute the STA of the MUA to map the RFs for each channel of the probe, and 
3. plot the estimated RFs.

**Software:** 
1. *Requirements:* Python 3, [PsychoPy toobox](https://www.psychopy.org/download.html)
2. An example of a short Neuropixels recording (5 GB) is provided as a reference for calibration purposes.

**Hardware:**  
To ensure proper alignment of the recorded neuronal activity to the timing of the visual stimulus frames, each frame of the stimulus matrix should be paired with a timestamp (TTL).
1. *Display of the visual stimuli:*  
install the PsychoPy toolbox and make sure that the numbers of the stimulus frames and the produced TTLs are matched. 
2. *Hardware options:* 
    1. for the script in its current state: the Stim TTL should simply be plugged into the front digital inputs of your Neuropixels card. 
    2. With additional NI-DAQ inputs in your NI-PXIe where your TTL should be recorded into the digital inputs number 3 (or change the "event_keys "frametimes" to your digital inputs). When using such an hardware settings an extra synchronization step between both card will be is required which will be activated by the options (`align_to_probe_timestamps=True` + write in your probe TTLs directors in "probe_ttl_dir="). 
