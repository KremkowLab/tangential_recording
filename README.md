# Probe placement optimization for Neuropixels tangential recordings

This repo implements the receptive field (RF) analysis via spike-triggered average (STA) of the multi-unit activity (MUA) from Neuropixels probe recordings. These scripts intend to quickly test the presence of any neuronal responses to visual stimuli in order to optimize the probe placement during tangential recordings in the mouse superior colliculus ([Sibille et al., 2021](https://www.biorxiv.org/content/10.1101/2021.06.12.448191v1.abstract)). Specifically, sparse-noise stimuli were presented to map the RFs. The peri-stimulus time histograms (PSTHs) and the corresponding RFs will be produced as part of the outputs.

Run the `MUA_STA_RFs_exec.py` to:
1. extract the MUA from the Neuropixels recording,
2. compute the STA of the MUA to map the RFs for each channel of the probe, and 
3. plot the estimated RFs.

**Software:** 
1. *Requirements:* Python 3, [PsychoPy toobox](https://www.psychopy.org/download.html)
2. An example of a [short Neuropixels recording](https://drive.google.com/drive/folders/19J-1CywCBwKaT7fIeQdrptI3LV4aR7e6?usp=sharing) (5 GB) is provided as a reference for calibration purposes.

**Hardware:**  
To ensure proper alignment of the recorded neuronal activity to the timing of the visual stimulus frames, each frame of the stimulus matrix should be paired with a stimulus TTL.
1. *Display of the visual stimuli:*  
install the PsychoPy toolbox and make sure that the numbers of the stimulus frames and the obtained stimulus TTLs are matched. 
2. *Hardware options:* 
    1. Connect the stimulus TTL to the front digital inputs of the Neuropixels card. Use `align_to_probe_timestamps=False` in the script.
    2. Connect the stimulus TTL to the additional NI-DAQ digital inputs in the NI-PXIe. This hardware setting requires an extra synchronization step between the two cards. In the script, set `align_to_probe_timestamps=True` and specify the probe event TTLs directory as `probe_ttl_dir`. 

*Make sure that the NI-DAQ hardware's digital input number `d` used for the stimulus TTLs matches the `channel state` of the `event_keys` named `"frametimes"` in the script.*

A test recording is available in: https://zenodo.org/record/6850116#.YtQNq4RBzQM
