#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:49:06 2018

@author: jenskremkow
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import time
# %%
date = '20190430'

y = 36
x = 22
n_trials = 10
targets_per_frame = 2
target_size = 3 # 1 = one position in the grid, 2 = 2x2 positions in the grid
target_size_offset = target_size-1
no_go_window = 6

x_with_offset = x + target_size_offset # this is because on the right border we have to 
y_with_offset = y + target_size_offset


n_pos = x_with_offset*y_with_offset*n_trials
n_frames = int((x_with_offset*y_with_offset*n_trials)/targets_per_frame)

background = 0.
target = 1.


stimulus = np.ones([y_with_offset,x_with_offset,n_frames+2000])*background# the extra 1000 are buffer


print(n_frames)
print(n_frames*(1/120.)*10/60.)
# %%
xs = np.arange(0,x_with_offset)
ys = np.arange(0,y_with_offset)

xs, ys = np.meshgrid(xs,ys)
xs = xs.flatten()
ys = ys.flatten()

xs = np.repeat(xs,n_trials)
ys = np.repeat(ys,n_trials)
#pol_D = np.ones(xs.shape)*-1.
#pol_L = np.ones(xs.shape)*1.


# %
# we have to copy this twice for having two polarities
#xs = np.append(xs,xs) # we have each position twice
#ys = np.append(ys,ys)
#pols = np.append(pol_D,pol_L)

# %

# np.random.seed(124322325)
#np.random.seed(1241961285) # works with 6
np.random.seed(4119) # works almost with 7
#np.random.seed(441961286) # works with 7



randseq = np.random.permutation(len(xs))# initial random positions

pos_x = xs[randseq]
pos_y = ys[randseq]
#pol_xy = pols[randseq]

# %


#plt.figure(1)
#plt.clf()

current_frame = 0
#for n in range(n_frames):
while len(pos_x) > 0:
    print('frame: '+str(current_frame)+'/'+str(n_frames)+' len pos:'+str(len(pos_x)))
    
    stimulus_frame_tmp = np.ones([y_with_offset,x_with_offset])*background
    stimulus_frame_no_go_area_tmp = np.zeros([y_with_offset,x_with_offset])
    
    # %
    # we take the first position from the remaining list of positions
    x_tmp = pos_x[0]
    y_tmp = pos_y[0]
    
    # %
    #target_size = 3
    #x_tmp = 44
    #y_tmp = 10
    
    #stimulus_frame_tmp = np.zeros([y_with_offset,x_with_offset])
    #stimulus_frame_no_go_area_tmp = np.zeros([y_with_offset,x_with_offset])
    
    
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
    #stimulus_frame_no_go_area_tmp[y_tmp_window_start:y_tmp_window_stop,x_tmp_window_start:x_tmp_window_stop] = 2.
    
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
                #pol_xy = pol_xy[randseq_tmp]
                
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
                #pol_xy = pol_xy[1:]
                # we increase the counter
                count += 1
                if count == targets_per_frame:
                    go = 0
                if len(pos_x) <= 0:
                    go = 0
        
    # we add current frame
    stimulus[:,:,current_frame] = stimulus_frame_tmp.copy()
    current_frame += 1

# %%
stimulus_frames = stimulus[target_size_offset:,target_size_offset:,0:current_frame]
plt.figure(5)    
plt.clf()

plt.subplot(2,1,1)
plt.pcolor(stimulus[:,:,current_frame-1])
plt.colorbar()  
plt.clim([-1,1])
   
plt.subplot(2,1,2)
plt.pcolor(stimulus_frames[:,:,-1])
plt.colorbar()  
plt.clim([-1,1]) 
    
# %%

#d = stimulus == -1.
#stimulus_l = stimulus.copy()
#stimulus_l[d] = 0.
#
#l = stimulus == 1.
#stimulus_d = stimulus.copy()
#stimulus_d[l] = 0.

plt.figure(4)    
plt.clf()

plt.subplot(1,1,1)
plt.pcolor(stimulus_frames.mean(2))
plt.colorbar()   

#plt.subplot(3,1,2)
#plt.pcolor(stimulus_l.mean(2))
#plt.colorbar()  
#
#plt.subplot(3,1,3)
#plt.pcolor(stimulus_d.mean(2))
#plt.colorbar() 



# %% we have to again randomize the stimulus, such that the last frames with the single pixels are not at the end
randseq = np.random.permutation(stimulus_frames.shape[2])# initial random positions

stimulus = stimulus_frames[:,:,randseq]

# %%

plt.figure(40)    
#plt.clf()

plt.subplot(1,1,1)
plt.pcolor(stimulus.mean(2))
plt.colorbar()   


# %%
# %%
#plt.close('all')
plt.figure(1)
plt.clf()
plt.ion()

for n in range(10):
    plt.pcolor(stimulus[:,:,n]) 
    plt.draw()
    plt.clim([-1.,1.])
    plt.pause(0.05)
    
plt.show()
    


# %% here we identify the ind of the targets
target_positions_x = {}
target_positions_y = {}


for i in range(stimulus.shape[2]):
    tmp = stimulus[:,:,i]
    ytmp,xtmp = np.where(tmp==target)
    target_positions_x[str(i)] = xtmp
    target_positions_y[str(i)] = ytmp
    #print(len(xtmp))
    

# %%
frames = stimulus.astype('int8')            

stimulus = {}
stimulus['frames'] = frames   
stimulus['target'] = target
stimulus['background'] = background
stimulus['target_positions'] = {'x':target_positions_x,'y':target_positions_y}

# %% SL
np.save('locally_light_sparse_noise_'+str(x)+'_'+str(y)+'_target_size_'+str(target_size)+'_targets_per_frame_'+str(targets_per_frame)+'_trials_'+str(n_trials)+'_background_'+str(background)+'_'+date,stimulus)



# %%        
# % SD
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

# % SD
np.save('locally_dark_sparse_noise_'+str(x)+'_'+str(y)+'_target_size_'+str(target_size)+'_targets_per_frame_'+str(targets_per_frame)+'_trials_'+str(n_trials)+'_background_'+str(background)+'_'+date,stimulus)
   

# %%
#plt.figure(40)    
#plt.clf()
#
#plt.subplot(1,1,1)
#plt.pcolor(stimulus[:,:,1])
#plt.colorbar()   
#plt.clim([-1,1.])
## %%
#plt.close('all')
#plt.figure(1)
#plt.clf()
#plt.ion()
#
#for n in range(10):
#    plt.pcolor(stimulus[:,:,n]) 
#    plt.draw()
#    plt.clim([-1.,1.])
#    plt.pause(0.05)
#    
#plt.show()
    

# %%
np.sum(stimulus_frames[1,1,:] == 1)
