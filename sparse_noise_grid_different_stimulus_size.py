#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:49:06 2018

@author: jenskremkow
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
date = '20190430'

y = 36
x = 22
n_trials = 10
targets_per_frame = 1
target_size = 1 # 1 = one position in the grid, 2 = 2x2 positions in the grid
target_size_offset = target_size-1
no_go_window = 6
x_with_offset = x + target_size_offset # this is because on the right border we have to 
y_with_offset = y + target_size_offset
n_pos = x_with_offset*y_with_offset*n_trials
n_frames = (x_with_offset*y_with_offset*n_trials)/targets_per_frame
n_frames = int(n_frames)
background = 0.
target = 1.
stimulus = np.ones([y_with_offset,x_with_offset,n_frames+2000])*background# the extra 2000 are buffer

# %%
xs = np.arange(0,x_with_offset)
ys = np.arange(0,y_with_offset)
xs, ys = np.meshgrid(xs,ys)
xs = xs.flatten()
ys = ys.flatten()
xs = np.repeat(xs,n_trials)
ys = np.repeat(ys,n_trials)

np.random.seed(4119)
randseq = np.random.permutation(len(xs))# initial random positions
pos_x = xs[randseq]
pos_y = ys[randseq]

current_frame = 0
while len(pos_x) > 0:
    print('frame: '+str(current_frame)+'/'+str(n_frames)+' len pos:'+str(len(pos_x)))
    
    stimulus_frame_tmp = np.ones([y_with_offset,x_with_offset])*background
    stimulus_frame_no_go_area_tmp = np.zeros([y_with_offset,x_with_offset])
    # we take the first position from the remaining list of positions
    x_tmp = pos_x[0]
    y_tmp = pos_y[0]
    # % we have to include the window now
    x_tmp_window_start = x_tmp 
    x_tmp_window_stop = x_tmp + target_size
    y_tmp_window_start = y_tmp 
    y_tmp_window_stop = y_tmp + target_size
    # we have to make sure that the window is in the stimulus
    x_tmp_window_start = np.max([x_tmp_window_start,0])
    y_tmp_window_start = np.max([y_tmp_window_start,0])
    x_tmp_window_stop = np.min([x_tmp_window_stop,x_with_offset])
    y_tmp_window_stop = np.min([y_tmp_window_stop,y_with_offset])
    stimulus_frame_tmp[y_tmp_window_start:y_tmp_window_stop,x_tmp_window_start:x_tmp_window_stop] = target # here the first pixel is in the box
    # we imprint the no go area
    x_nogo_start = x_tmp - no_go_window
    y_nogo_start = y_tmp - no_go_window
    x_nogo_stop = x_tmp + no_go_window+1+ target_size
    y_nogo_stop = y_tmp + no_go_window+1+ target_size
    # we have to make sure that the window is in the stimulus
    x_nogo_start = np.max([x_nogo_start,0])
    y_nogo_start = np.max([y_nogo_start,0])
    x_nogo_stop = np.min([x_nogo_stop,x_with_offset])
    y_nogo_stop = np.min([y_nogo_stop,y_with_offset])
    stimulus_frame_no_go_area_tmp[y_nogo_start:y_nogo_stop,x_nogo_start:x_nogo_stop] = 1    
    # now we remove that pos from the list
    pos_x = pos_x[1:]
    pos_y = pos_y[1:]
        
    if len(pos_x) > 0:
        # now we have to look for other pixel that are not close by
        go = 1
        count = 1 # we start with one beacuse the first target was placed already
        count_loops = 0
        count_got_stuck_in_frame = 0
        while go:
            count_loops += 1
            #    # we take the first of the list
            x_tmp = pos_x[0].copy()
            y_tmp = pos_y[0].copy()
            if (stimulus_frame_no_go_area_tmp[y_tmp,x_tmp] == 1) & (count_got_stuck_in_frame <= 500):
                # yes we are in the no go... we just shuffel the data and repeat
                # we try a few times
                randseq_tmp = np.random.permutation(len(pos_x))#
                pos_x = pos_x[randseq_tmp]
                pos_y = pos_y[randseq_tmp]                
                count_got_stuck_in_frame += 1
            elif (stimulus_frame_no_go_area_tmp[y_tmp,x_tmp] == 1) & (count_got_stuck_in_frame > 500):
                # if we fail we just make a new frame
                # we do that by leaving thw while loop. This will not remove the current pos from the list
                go = 0
            else:
                # hurra!
                # % we have to include the window now
                x_tmp_window_start = x_tmp 
                x_tmp_window_stop = x_tmp + target_size
                y_tmp_window_start = y_tmp 
                y_tmp_window_stop = y_tmp + target_size
                # we have to make sure that the window is in the stimulus
                x_tmp_window_start = np.max([x_tmp_window_start,0])
                y_tmp_window_start = np.max([y_tmp_window_start,0])
                x_tmp_window_stop = np.min([x_tmp_window_stop,x_with_offset])
                y_tmp_window_stop = np.min([y_tmp_window_stop,y_with_offset])
                stimulus_frame_tmp[y_tmp_window_start:y_tmp_window_stop,x_tmp_window_start:x_tmp_window_stop] = target # here the first pixel is in the box
                # we imprint the no go area
                x_nogo_start = x_tmp - no_go_window
                y_nogo_start = y_tmp - no_go_window
                x_nogo_stop = x_tmp + no_go_window+1+ target_size
                y_nogo_stop = y_tmp + no_go_window+1+ target_size
                # we have to make sure that the window is in the stimulus
                x_nogo_start = np.max([x_nogo_start,0])
                y_nogo_start = np.max([y_nogo_start,0])
                x_nogo_stop = np.min([x_nogo_stop,x_with_offset])
                y_nogo_stop = np.min([y_nogo_stop,y_with_offset])
                stimulus_frame_no_go_area_tmp[y_nogo_start:y_nogo_stop,x_nogo_start:x_nogo_stop] = 1
                # we remove the current pos from the list
                pos_x = pos_x[1:]
                pos_y = pos_y[1:]
                count += 1
                if count == targets_per_frame:
                    go = 0
                if len(pos_x) <= 0:
                    go = 0
        
    # we add current frame
    stimulus[:,:,current_frame] = stimulus_frame_tmp.copy()
    current_frame += 1

# %% here we identify the ind of the targets
target_positions_x = {}
target_positions_y = {}
for i in range(stimulus.shape[2]):
    tmp = stimulus[:,:,i]
    ytmp,xtmp = np.where(tmp==target)
    target_positions_x[str(i)] = xtmp
    target_positions_y[str(i)] = ytmp

# %% SL
frames = stimulus.astype('int8')
stimulus = {}
stimulus['frames'] = frames   
stimulus['target'] = target
stimulus['background'] = background
stimulus['target_positions'] = {'x':target_positions_x,'y':target_positions_y}
np.save('locally_light_sparse_noise_'+str(x)+'_'+str(y)+'_target_size_'+str(target_size)+'_targets_per_frame_'+str(targets_per_frame)+'_trials_'+str(n_trials)+'_background_'+str(background)+'_'+date,stimulus)

# %% SD
frames = frames.astype('double')
frames = frames *-1.
frames = frames.astype('int8')   
target = target *-1.
background = background *-1.
stimulus = {}
stimulus['frames'] = frames   
stimulus['target'] = target
stimulus['background'] = background
stimulus['target_positions'] = {'x':target_positions_x,'y':target_positions_y}
np.save('locally_dark_sparse_noise_'+str(x)+'_'+str(y)+'_target_size_'+str(target_size)+'_targets_per_frame_'+str(targets_per_frame)+'_trials_'+str(n_trials)+'_background_'+str(background)+'_'+date,stimulus)
