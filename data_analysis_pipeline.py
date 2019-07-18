import data_analysis as da
import matplotlib.pyplot as plt
from copy import deepcopy
import mne

#%% Initialize subject to analyze

subj = 'HeBoDLVR1'                                    # name of subject
# location = 'R:\\NeurOne\\NeurOneUser\\'             # if data is on server
location = 'D:\\Dirk\\NeurOne\\NeurOneUser'         # if data is on local computer
folder = 'subj1\\'                                    # folder that contains .xdf files
path = location + '\\' + subj + '\\' + folder         # creates full path

#%% I. Load data

raw = da.xdf2raw(path, subj)

#%% create copy of original raw objects for PSD

raw_BAK = deepcopy(raw)   # backup of raw data for PSD

#%% II. Resample

sfreq_new = 250                          # desired sample rate

raw = da.resample(raw, sfreq_new)

#%% III. Comb filter 50Hz and 90Hz and 52.1Hz

freqs = [50, 90, 52.1]          # frequencies (+ harmonics) at which notch filter ist applied

raw = da.comb_filt(raw, freqs)

#%% IV. Remove DC offset

raw, offsets = da.remove_offset(raw)

#%% Concatenate raws

raw_conc = da.concat_raws(raw)

#%% V. Raw data viewer

raw_conc.plot(remove_dc=False, scalings='auto', n_channels=75, events=raw_conc.events, event_id=raw_conc.event_id,
              title=subj)

#%% VI. PSD topo

fmax = 250                # end frequency to consider
blocks = [1, 2, 10]   # blocks for which to plot PSDs

da.plot_raw_psd(raw_BAK, fmax, blocks)

#plt.gcf().axes[0].set_xscale('symlog')
#plt.gcf().axes[0].set_yscale('symlog')

#%% VII. variance topo

percentile = 95                 # percentile specifying the upper bound of the color range

da.plot_var_topo(raw, percentile, as_log=True)

#%% VII. Epoch data

event_ids = ['Monster active', 'Monster left', 'Monster right']              # Epochs of these events will be extracted
tmin = -0.5                                  # The time in seconds before the event, in which the EEG Data is extracted.
tmax = 4                                      # The time in seconds after the event, in which the EEG Data is extracted.

# epoch data for each block separately
epochs = da.extract_epochs(raw, event_ids, tmin, tmax)

# concatenate epochs of same events of different blocks
epochs_conc = da.concat_epochs(epochs, event_ids)

#%% using mne functions to plot epochs
# can be used to drop epochs

# plot raw epoch data of event specified by index
epochs_conc[1].plot(scalings='auto')

#epochs_conc[0].plot_psd(picks=['all'])


#%% Average epochs with same events

evoked, evo_se = da.average_epochs(epochs_conc, error='std')


#%% Pot evoked

mne.viz.plot_evoked_topo(evoked, scalings=1)  # topographically plot averaged ERPs with one line per marker

# IX
da.plot_evo_topo(evoked, evo_se)              # topographically plot averaged ERPs with STD/SEM

da.plot_evo(evoked)

# some additional mne plot functions:
da.plot_evo_joint(evoked, times=[-0.05, 0, 0.2, 0.3, 0.4])

mne.viz.plot_evoked_topomap(evoked[0])

evoked[1].plot(spatial_colors=True, gfp=True, picks='all')
evoked[1].plot_joint(times=[.17, .28, .36, .5])

#%% plot PSDs

#plot PSDs
# X
da.plot_epoch_psd(epochs_conc, tmin=-0.5, tmax=2)           #PSDs over all blocks
# X
da.plot_block_psd(epochs, block=16, tmin=-0.5, tmax=1)      #PSD for specific block

#%% plot TFR

# XIV
da.plot_tfr(epochs_conc, event=1)           # baseline correction as specified for mne.time_frequency.AverageTFR.plot_topo() from -500 ms to -0 ms before onset

# XIV
da.epoch_tfr(epochs, events=[1,2]) # baseline correction: Calculate baseline (-500 ms - 0 ms before marker, median across time & trials in two steps, all markers pooled)

# XV
da.plot_rel_tfr(epochs_conc, events=[1,2])  # power event1/ power event2, baseline correction as specified for mne.time_frequency.AverageTFR.plot_topo() from -500 ms to -0 ms before onset




