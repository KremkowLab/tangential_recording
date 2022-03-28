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
To ensure proper alignment of the recorded neuronal activity to the timing of the visual stimulus frames, each frame of the stimulus matrix should be paired with a stimulus TTL.
1. *Display of the visual stimuli:*  
install the PsychoPy toolbox and make sure that the numbers of the stimulus frames and the obtained stimulus TTLs are matching. 
2. *Hardware options:* 
    1. Simply plug the stimulus TTL cable into the front digital inputs of your Neuropixels card. Use `align_to_probe_timestamps=False` in the script.
    2. With additional NI-DAQ inputs in the NI-PXIe, an extra synchronization step between the two cards is required. In the script, simply use `align_to_probe_timestamps=True` and specify the probe event TTLs directory as `probe_ttl_dir`.
3. Ensure that the hardware's digital input number `d` used for recording the stimulus TTLs matches the `"frametimes"`'s `channel state` of the `event_keys` in the script.
