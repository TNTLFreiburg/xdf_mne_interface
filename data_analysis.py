from importlib import reload

import mne_interface as mif
import matplotlib.pyplot as plt
import numpy as np
import mne
import os
import sys
import resampy
from copy import deepcopy
from natsort import natsorted
import re


#%% Can be used to to disable print outputs of functions if annoying

# Disable
def blockprint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enableprint():
    sys.stdout = sys.__stdout__


#%% I. Load data

def xdf2raw(path,subj):
    """ Loads all EEG files (.xdf) from specified path and stores them in a list of mne.raw objects.
        Converting from xdf to mne is done by mne_interface.xdf_loader().

        Args:
            path : The location to the .xdf files from which the data is extracted.
            subj : name of subject

        Returns:
            raw: A list of mne raw objects.

    """

    blocks = natsorted(os.listdir(path))                            # sorted list of .xdf files
    raw = []
    for run_idx in range(len(blocks[0::])):
        print('Loading file >>> ' + blocks[run_idx], end='\n')
        blockprint()                                                # blocks print statements of xdf_loader
        raw_temp = mif.xdf_loader(path + blocks[run_idx])
        raw_temp._filenames = [blocks[run_idx]]
        raw_temp.info['subject_info'] = subj
        raw.append(raw_temp)
        enableprint()
        del raw_temp
    return (raw)


#%% II. Resample


def resample(raw, sfreq_new):
    """ Downsamples list of mne.raw objects to desired sampling rate using resampy.resample()
        Events and time vectors also get updated to new sampling rate

        Args:
                raw: list of mne.raw objects
                sfreq_new: desired sample rate

        Returns:
                raw: A list of the downsampled mne raw objects.
    """

    for run_idx in range(len(raw)):
        print('Resampling file >>> '+str(run_idx+1)+'/'+str(len(raw)), end='\r')

        raw[run_idx].events[:, 0] = raw[run_idx].events[:, 0] / (raw[run_idx].info['sfreq']/sfreq_new)  # "downsamples" timestamps of events

        raw[run_idx]._data = resampy.resample(raw[run_idx]._data,                                       # downsampling data
                                              sr_orig=raw[run_idx].info['sfreq'],
                                              sr_new=sfreq_new,
                                              filter='kaiser_fast')

        raw[run_idx]._times = raw[run_idx]._times[0::int(raw[run_idx].info['sfreq']/sfreq_new)]   # takes every n_th timepoint (n = sfreq_original / sfreq_new)
        raw[run_idx].info['sfreq'] = sfreq_new
        raw[run_idx]._last_samps[0] = raw[run_idx]._data.shape[1]-1         # updates raw._last_samps to correctly display object info line
    return (raw)


#%% III. Comb filter 50Hz and 90Hz and 52.1Hz

def comb_filt(raw, freqs):
    """ Applies a combfilter to a list of raw objects using mne.raw.notch_filter. Signal is filtered at the given
        frequencies and their harmonics

        Args:
                raw: list of mne.raw objects
                freqs: a list of frequencies
        Returns:
                raw: list of filtered raw objects

        """
    for run_idx in range(len(raw)):
        for f in freqs:
            print('############## Filtering block >>> ' + str(run_idx + 1) + '/' + str(len(raw)), end='\n')
            raw[run_idx].notch_filter(np.arange(f, raw[run_idx].info['sfreq']/2, f), trans_bandwidth=0.4, phase='zero-double',  n_jobs='cuda')
    return raw

#%% IV. Remove DC offset


def remove_offset(raw):
    """ Subtracts the mean from the signal and stores offset values

            Args:
                    raw: list of mne.raw objects
            Returns:
                    raw: A list of the mne.raw.objects with offset removed
                    offsets: a list of offsets (n_blocks x n_channels)
            """
    offsets = []
    for run_idx in range(len(raw)):
        offset = (np.mean(raw[run_idx]._data, axis=1))
        offsets.append(offset)
        raw[run_idx]._data = (raw[run_idx]._data.transpose() - offsets[run_idx]).transpose()
    return (raw, offsets)

#%% V. Concatenate raws for raw data viewer

# create concatenated raw
def concat_raws(raw):
    """ Concatenate raws with events

                Args:
                        raw: list of mne.raw objects
                Returns:
                        raw_conc: the concatenated raw object with concatenated events
                """
    events = [r.events for r in raw]
    raw_conc, events_conc = mne.concatenate_raws(deepcopy(raw), events_list=events)
    raw_conc.events = events_conc
    return (raw_conc)

#%% VI. PSD topo


def plot_raw_psd(raw_BAK, fmax, blocks):
    """ topographically plots PSDs for EEG channels of specified Blocks. Each block will be plotted in a separat figure.
        Uses mne.viz.plot_raw_psd_topo() which uses WELCH method for FFT.

                    Args:
                            raw_BAK: List of unprocessed raw objects.
                            fmax: End frequency to consider.
                            blocks: Blocks of interest to plot PSD
                    """
    montage = mne.channels.read_montage("standard_1005", ch_names=raw_BAK[0].info["ch_names"])
    raw_BAK[0].set_montage(montage)

    for block_idx in blocks:
        block_idx = block_idx-1
        mne.viz.plot_raw_psd_topo(raw_BAK[block_idx], fmax=fmax, n_fft=int(raw_BAK[block_idx].info['sfreq']*10),
                                  show=True, n_jobs=1, color='w', fig_facecolor='k', axis_facecolor='k')
        plt.title(raw_BAK[block_idx].info['subject_info'] + ', ' + raw_BAK[block_idx].filenames[0], color='k')


#%% VII. variance topo


def plot_var_topo(raw, percentile):
    """ topographically plots variance of EEG channels for each block.

                        Args:
                                raw: List raw objects.
                                percentile: specifying the upper bound of the color range. Percentile of all blocks and all channels
                        """
    vars = []
    # compute variances for each channel and file
    for run_idx in range(len(raw)):
        var = np.var(raw[run_idx]._data, axis=1)
        vars.append(var)
        raw[run_idx]._var = var

    perc = np.percentile(vars, percentile)

    eeg_layout = mne.channels.make_eeg_layout(raw[run_idx].info)
    for run_idx in range(len(raw)):

        plt.subplot(4, 4, run_idx+1)
        ax = mne.viz.plot_topomap(raw[run_idx]._var[0:eeg_layout.ids.size], pos=eeg_layout.pos, names=eeg_layout.names,
                                  show_names=False, outlines='head', extrapolate='local', contours=0, vmax=perc)

        plt.title(raw[run_idx]._filenames[0])
        if run_idx+1 == len(raw):
            plt.colorbar(ax[0])
    plt.gcf().suptitle('Variances, ' + raw[0].info['subject_info'])

#%% VII. Epoch data


def extract_epochs(raw, event_ids, tmin, tmax):
    """ Extracts and accumulates given timeframes around events from a list of raw.objects and stores them in a list of
        epochs.objects for each block which get stored in a list of events with the same order as specified in event_ids.
    Args:
        raw: a list of raw objects

        event_ids: a list of strings with the names of the events to be extracted

        tmin: The time in seconds before the event, in which the EEG Data is extracted.

        tmax: The time in seconds after the event, in which the EEG Data is extracted.

    Returns:
        epochs: lists of mne Epochs objects for each block stored in list for each event
    """
    epochs = []
    for event_id in event_ids:
        print('#######################################'+event_id)
        master_id = {}
        epoch = []
        master_legend = []

        for run_idx in range(len(raw)):
            print('####################################### block' + str(run_idx))
            current_raw = raw[run_idx]
            current_events = current_raw.events
            current_id = current_raw.event_id

            # Compute which actions are available in the current file
            here = np.array([bool(re.search(event_id, element)) for element in list(current_id.keys())])
            legend = np.array(list(current_id.keys()))[here]

            # Update Master legend and ID if the current file includes new actions
            for event in legend[[item not in master_legend for item in legend]]:
                master_id[event] = len(master_id)
                master_legend = np.append(master_legend, event)

            picked_events = np.empty([0, 3], dtype=int)
            picked_id = {}

            for this in legend:
                # Get all appropriate events
                picked_events = np.append(picked_events, current_events[current_events[:, 2] == current_id[this]], axis=0)
                # Update the ID according to master
                picked_events[:, 2][picked_events[:, 2] == current_id[this]] = master_id[this]
                # Build up a temp ID dict for the current Epochs
                picked_id[this] = master_id[this]

            # Building empty Epochs will throw errors
            if not picked_id:
                continue
            current_epoch = mne.Epochs(current_raw, picked_events, picked_id, tmin=tmin, tmax=tmax, baseline=(None, 0),
                                       detrend=0, preload=True)
            current_epoch._filename = current_raw._filenames
            current_epoch.load_data()

            # Append the current epochs if there are epochs to append to
            if not epoch:
                epoch.append(current_epoch.copy())
            else:
                epoch.append(mne.Epochs(current_raw, picked_events, picked_id, tmin=tmin, tmax=tmax, baseline=(None, 0),
                                        detrend=0, preload=True))
                epoch[run_idx]._filename = current_raw._filenames

        epochs.append(epoch)
    return epochs

#%%
def concat_epochs(epochs, event_ids):
    """ concatenate epochs of same events of different blocks
        Args:
            epochs: a list of lists of epoch objects

            event_ids: a list of strings with the names of the events to be extracted

        Returns:
            epochs_conc: lists of concatenated mne Epochs objects for each event
        """
    epochs_conc = []
    for event_id in range(len(event_ids)):
        epochs_conc.append(mne.epochs.concatenate_epochs(epochs[event_id], add_offset=True))
    return epochs_conc


#%% Average epochs with same events

def average_epochs(epochs_conc, error='std'):
    """ average epochs of same event type to create evoked object. Also creates evoked object with std/sem values instead of mean.
            Args:
                epochs: a list of concatenated epoch objects

                error: type of error to be computed. 'std' - standard deviation, 'sem' - standard error of the means. Default = 'std'

            Returns:
                evoked: evoked objects for each event stored in a list
                evo_se: evvoked objects with errors for each event stored in a list
            """
    evoked = []
    evo_se =[]
    std_dev = lambda x: np.std(x, axis=0)

    for ep_idx in range(len(epochs_conc)):
        evoked.append(mne.Epochs.average(epochs_conc[ep_idx], picks='all'))

        if error =='sem':
            evo_se.append(mne.Epochs.average(epochs_conc[ep_idx], picks='data', method='std'))   # standard error
            evo_se[ep_idx].error = 'SEM'
        elif error == 'std':
            evo_se.append(mne.Epochs.average(epochs_conc[ep_idx], picks='data', method=std_dev))   # standard deviation
            evo_se[ep_idx].error = 'STD'
    return evoked, evo_se


#%% plot evoked

def plot_evo(evoked):
    for event_id in range(len(evoked)):
        ax =plt.subplot(len(evoked), 1, event_id+1)
        evoked[event_id].plot(spatial_colors=True, gfp=True, picks='eeg', axes=ax, titles=evoked[event_id].comment, scalings=1)

def plot_evo_joint(evoked, times):
    for event_id in range(len(evoked)):

        evoked[event_id].plot_joint(times=times, picks='eeg', title=evoked[event_id].comment,
                                    topomap_args={'outlines':'head'},
                                    ts_args={})


#%% plot evoked

def plot_evo_topo(evoked, evo_se):
    """ topographically plots ERPs
        Args:
            evoked: a list of evoked objects

            evo_se: a list of evoked objects with deviation/error values
    """
    f = plt.figure()

    legends = []
    y_max = []
    y_min = []
    for evo_idx in range(len(evoked)):
        evoked[evo_idx].pick_types(meg=False, eeg=True)
        y_max.append(np.max(evoked[evo_idx]._data))
        y_min.append(np.min(evoked[evo_idx]._data))
        #y_max.append(np.max((evoked[evo_idx]._data + evo_se[evo_idx]._data)))
        #y_min.append(np.min((evoked[evo_idx]._data - evo_se[evo_idx]._data)))


    def my_callback(ax, ch_idx):
        """
        This block of code is executed once you click on one of the channel axes
        in the plot. To work with the viz internals, this function should only take
        two parameters, the axis and the channel or data index.
        """
        for evo_idx in range(len(evoked)):
            ax.vlines([0], np.min(y_min), np.max(y_max), linestyles='dashed')
            ax.plot(evoked[evo_idx].times, evoked[evo_idx]._data[ch_idx])
            plt.fill_between(evoked[evo_idx].times,
                             evoked[evo_idx]._data[ch_idx]+evo_se[evo_idx]._data[ch_idx],
                             evoked[evo_idx]._data[ch_idx]-evo_se[evo_idx]._data[ch_idx], alpha=.1)
            plt.xlim([evoked[evo_idx].times[0],evoked[evo_idx].times[-1]])
            plt.ylim([np.min(y_min), np.max(y_max)])
            plt.legend(legends[0:len(evoked)])
            plt.xlabel('Time (s)')
            plt.ylabel(r'$\mu$V')


    for ax, idx in mne.viz.iter_topography(evoked[0].info,
                                           fig_facecolor='white',
                                           axis_facecolor='white',
                                           axis_spinecolor='white',
                                           on_pick=my_callback,
                                           fig=f):
        for evo_idx in range(len(evoked)):
            legends.append(evoked[evo_idx].comment)
            ax.vlines([0], np.min(y_min), np.max(y_max), linestyles= 'dashed', linewidth=0.5)
            ax.plot(evoked[evo_idx].times, evoked[evo_idx]._data[idx], linewidth=0.5)
            plt.fill_between(evoked[evo_idx].times,
                             evoked[evo_idx]._data[idx] + evo_se[evo_idx]._data[idx],
                             evoked[evo_idx]._data[idx] - evo_se[evo_idx]._data[idx], alpha=.1)
            plt.ylim([np.min(y_min), np.max(y_max)])
            plt.xlim([evoked[evo_idx].times[0], evoked[evo_idx].times[-1]])
        plt.gcf().suptitle('Average ERPs ' + r'$\pm$ ' + evo_se[0].error + ', ' + evoked[0].info['subject_info'])
        #                   ha='right')


#%% epoch PSD


def plot_epoch_psd(epochs, epo_list=[0,1,2], fmax=np.inf , tmin=-0.5, tmax=1):
    """ topographically plots PSD for the averaged epochs
        Args:
             epochs: a list the concatenated epochs objects
             epo_list: list of indices (=events) to compare
             tmin: time before event
             tmax: time after event
    """
    # epochs.pick_types(meg=False, eeg=True)
    f = plt.figure()
    sign_length = int(epochs[0].info['sfreq']/2+1)
    means = np.zeros((len(epochs), len(epochs[0]._channel_type_idx['eeg']), sign_length))
    stds = np.zeros((len(epochs), len(epochs[0]._channel_type_idx['eeg']), sign_length))
    legends = []

    def my_callback(ax, ch_idx):
        """
        This block of code is executed once you click on one of the channel axes
        in the plot. To work with the viz internals, this function should only take
        two parameters, the axis and the channel or data index.
        """
        for epo_idx in epo_list:
            ax.plot(freqs, means[epo_idx][ch_idx])
            #plt.fill_between(freqs, means[epo_idx][ch_idx]+stds[epo_idx][ch_idx], means[epo_idx][ch_idx]-stds[epo_idx][ch_idx], alpha=.1)
            plt.xlim([freqs[0], freqs[-1]])
            plt.legend(legends)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power (dB)')

    for epo_idx in epo_list:
        epo = epochs[epo_idx]
        epo.pick_types(meg=False, eeg=True)

        psds, freqs = mne.time_frequency.psd_welch(epo, n_fft=int(epo.info['sfreq']), n_per_seg=sign_length, fmax=fmax, tmin=tmin, tmax=tmax)

        psds = 20 * np.log10(psds)

        means[epo_idx] = np.median(psds, axis=0)
        stds[epo_idx] = np.std(psds, axis=0)
        legends.append(list(epochs[epo_idx].event_id.keys())[0])


    for ax, idx in mne.viz.iter_topography(epo.info,
                                           fig_facecolor='white',
                                           axis_facecolor='white',
                                           axis_spinecolor='white',
                                           on_pick=my_callback,
                                           fig=f):
        for epo_idx in epo_list:
            ax.plot(freqs, means[epo_idx][idx])
            plt.xlim([freqs[0], freqs[-1]])

        plt.gcf().suptitle('Average PSD, ' + str(tmin) + ' - ' + str(tmax) + ' sec, ' + epochs[0].info['subject_info'],
                           ha='right')
        plt.show()

#%% epoch PSD for each block


def plot_block_psd(epochs, block, epo_list=[0,1,2], fmax=np.inf , tmin=-0.5, tmax=1):
    """ topographically plots PSD for the averaged epochs for each block separately
            Args:
                 epochs: a list the concatenated epochs objects
                 block: number of block to plot
                 epo_list: list of indices (=events) to compare
                 tmin: time before event
                 tmax: time after event
        """
    #epochs.pick_types(meg=False, eeg=True)
    block = block-1
    f = plt.figure()
    sign_length = int(epochs[0][block].info['sfreq']/2+1)
    means = np.zeros((len(epochs), len(epochs[0][block]._channel_type_idx['eeg']), sign_length))
    stds = np.zeros((len(epochs), len(epochs[0][block]._channel_type_idx['eeg']), sign_length))
    legends = []

    def my_callback(ax, ch_idx):
        """
        This block of code is executed once you click on one of the channel axes
        in the plot. To work with the viz internals, this function should only take
        two parameters, the axis and the channel or data index.
        """
        for epo_idx in epo_list:
            ax.plot(freqs, means[epo_idx][ch_idx])
            #plt.fill_between(freqs, means[epo_idx][ch_idx]+stds[epo_idx][ch_idx], means[epo_idx][ch_idx]-stds[epo_idx][ch_idx], alpha=.1)
            plt.xlim([freqs[0], freqs[-1]])
            plt.legend(legends)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power (dB)')

    for epo_idx in epo_list:
        epo = epochs[epo_idx][block]
        epo.pick_types(meg=False, eeg=True)

        psds, freqs = mne.time_frequency.psd_welch(epo, n_fft=int(epo.info['sfreq']), n_per_seg=sign_length, fmax=fmax, tmin=tmin, tmax=tmax)

        psds = 20 * np.log10(psds)

        means[epo_idx] = np.median(psds, axis=0)
        stds[epo_idx] = np.std(psds, axis=0)
        legends.append(list(epochs[epo_idx][block].event_id.keys())[0])


    for ax, idx in mne.viz.iter_topography(epo.info,
                                           fig_facecolor='white',
                                           axis_facecolor='white',
                                           axis_spinecolor='white',
                                           on_pick=my_callback,
                                           fig=f):
        for epo_idx in epo_list:
            ax.plot(freqs, means[epo_idx][idx])
            plt.xlim([freqs[0], freqs[-1]])

        plt.gcf().suptitle('Average PSD, block#' + str(block+1) + ', ' + str(tmin) + ' - ' + str(tmax) + ' sec, ' + epochs[0][0].info['subject_info'],
                           ha='right')
        plt.show()


#%% Plot TFR

# baseline correction as specified for mne.time_frequency.AverageTFR.plot_topo()
def plot_tfr(epochs,event):
    freqs = np.arange(1,101,1)
    #freqs = np.logspace(*np.log10([1, 100]), num=90)

    n_cycles = freqs / 2  # different number of cycle per frequency

    power, itc = mne.time_frequency.tfr_morlet(epochs[event], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                               return_itc=True, decim=1, n_jobs=8)

    power.plot_topo(baseline=(-.5, 0), mode='logratio', title='Average power: Event - '+list(epochs[event].event_id.keys())[0])

    return power


# baseline correction: Calculate baseline (-500 ms - 0 ms before marker, median across time & trials in two steps, all markers pooled)
def epoch_tfr(epochs,events):

    freqs = np.arange(1,101,1)
    #freqs = np.logspace(*np.log10([1, 100]), num=90)
    n_cycles = freqs / 2  # different number of cycle per frequency

    TFR = []
    for eve_idx in events:
        powers = []
        TFR.append(powers)
        for block_idx in range(len(epochs[0])):

            powers.append(mne.time_frequency.tfr_morlet(epochs[eve_idx][block_idx], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                                       return_itc=False, decim=1, n_jobs=8, average=True))

    bl_median_time = []
    for block_idx in range(len(epochs[0])):
        bl_median_time.append(np.median(TFR[0][block_idx]._data[:, :, 0:np.int(0.5*250)],axis=2))
        bl_median_time.append(np.median(TFR[1][block_idx]._data[:, :, 0:np.int(0.5*250)], axis=2))

    baseline = np.median(bl_median_time,axis=0)

    TFR_bl_div = deepcopy(TFR)
    for eve_idx in range(len(events)):
        for block_idx in range(len(epochs[0])):
            for ch_idx in range(np.shape(baseline)[0]):
                for freq_idx in range(np.shape(baseline)[1]):
                    TFR_bl_div[eve_idx][block_idx]._data[ch_idx,freq_idx,:] = np.log(TFR[eve_idx][block_idx]._data[ch_idx,freq_idx,:] / baseline[ch_idx,freq_idx])

    #TFR_bl_median = deepcopy(TFR)
    #for eve_idx in events:
    #    for block_idx in range(len(epochs[0])):
    #       for ch_idx in range(np.shape(baseline)[0]):
    #            for freq_idx in range(np.shape(baseline)[1]):

    TFR_left = mne.time_frequency.tfr.combine_tfr(TFR_bl_div[0], weights='equal')
    TFR_right = mne.time_frequency.tfr.combine_tfr(TFR_bl_div[1], weights='equal')

    TFR_left.plot_topo(baseline=None, mode='logratio',
                    title='Monster left')
    TFR_right.plot_topo(baseline=None, mode='logratio',
                    title='Monster right')


#%%
# relative TFR
# power event1/ power event2, baseline correction as specified for mne.time_frequency.AverageTFR.plot_topo()
def plot_rel_tfr(epochs,events):
    freqs = np.arange(1,101,1)
    #freqs = np.logspace(*np.log10([1, 100]), num=90)

    n_cycles = freqs / 2  # different number of cycle per frequency

    power1, itc1 = mne.time_frequency.tfr_morlet(epochs[events[0]], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                                 return_itc=True, decim=1, n_jobs=8)
    power2, itc2 = mne.time_frequency.tfr_morlet(epochs[events[1]], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                                 return_itc=True, decim=1, n_jobs=8)

    rel_power = deepcopy(power1)
    rel_power._data = power1._data/power2._data

    rel_power.plot_topo(baseline=None, mode='logratio', title='relative average power: Event - '+list(epochs[events[0]].event_id.keys())[0]+'/'+list(epochs[events[1]].event_id.keys())[0], )
