import numpy as np
import xdf
import matplotlib.pyplot as plt
import re
import resampy
import mne
import cv2
from braindecode.datautil.signalproc import exponential_running_standardize, exponential_running_demean
from scipy.signal import filtfilt, iirnotch, butter
from processors import StandardizeProcessor


def xdf_loader(xdf_file):
    """ Loads an appropriate EEG file into a mne raw object.
    
    Args:
        xdf_file: The path to an .xdf file from which the data is extracted.
        
    Returns:
        raw: The mne raw object.
    
    """
    
    # Load the xdf file
    stream = xdf.load_xdf(xdf_file, verbose = False)
    
    # Extract the necessary event and eeg information
    stream_names = np.array([item['info']['name'][0] for item in stream[0]])
    game_state = list(stream_names).index('Game State')
    eeg_data = list(stream_names).index('NeuroneStream')

    sfreq = int(stream[0][eeg_data]['info']['nominal_srate'][0])
    game_state_series = np.array([item[0] for item in stream[0][game_state]['time_series']])
    game_state_times = np.array(stream[0][game_state]['time_stamps'])
    
    times = stream[0][eeg_data]['time_stamps']
    data = stream[0][eeg_data]['time_series'].T
    
    if (len(data) == 72):
        ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'emg', 'emg', 'emg', 'emg', 'eog', 'eog', 'eog', 'eog',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg']

        ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
                'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5',
                'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1',
                'Oz', 'O2', 'EMG_RH', 'EMG_LH', 'EMG_RF', 'EMG_LF', 'EOG_R', 'EOG_L', 'EOG_U', 'EOG_D',
                'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz',
                'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1',
                'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8',
                'PO7', 'PO8']
        
    elif (len(data) == 75):
        ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'emg', 'emg', 'emg', 'emg', 'eog', 'eog', 'eog', 'eog',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                'eeg', 'eeg', 'ecg', 'bio', 'bio']

        ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
                'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5',
                'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1',
                'Oz', 'O2', 'EMG_RH', 'EMG_LH', 'EMG_RF', 'EMG_LF', 'EOG_R', 'EOG_L', 'EOG_U', 'EOG_D',
                'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz',
                'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1',
                'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8',
                'PO7', 'PO8', 'ECG', 'Respiration', 'GSR']
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    raw = mne.io.RawArray(data, info)
    
    events = np.zeros([game_state_series.size, 3], 'int')
    
    # Calculate the closest frame in the EEG data for each event time
    events[:,0] = [np.argmin(np.abs(times-event_time)) for event_time in game_state_times]
    
    legend = np.unique(game_state_series)
    class_vector = np.zeros(game_state_series.size, dtype='int')
    event_id = {}
    for ii in np.arange(legend.size):
        class_vector += (game_state_series == legend[ii])*ii
        event_id[legend[ii]] = ii
        
    events[:,2] = class_vector
    
    # This will not get saved to .fif and has to be worked around
    raw.events = events
    raw.event_id = event_id

    # Set eeg sensor locations
    raw.set_montage(mne.channels.read_montage("standard_1005", ch_names=raw.info["ch_names"]))
    
    # Reference to the common average
    #raw.set_eeg_reference()
    
    return(raw)

def synchronize_video(xdf_file, vid):
    """ Predicts LSL time for every frame in a corresponding video. For use in a Notebook, preceed the command with "%matplotlib qt".
    
    Args:
        xdf: Path to the .xdf file of the experiment.
        vid: Path to the video file of the experiment
        
    Returns:
        frame_time: Array of LSL times. Index corresponds to frame in the video.
    
    """
    
    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()
    
    #
    f,a = plt.subplots()
    a.imshow(frame, cmap = 'gray')
    pos = []
    
    print('Click on the indicator and close the image.')
    
    def onclick(event):
        pos.append([event.xdata,event.ydata])
        
    f.canvas.mpl_connect('button_press_event', onclick)
    f.suptitle('Click on the indicator')
    
    plt.show(block = True)
    pos = np.array(pos[-1]).round().astype(int)
    print('Read pixel: ', pos)

    print('Start reading video')
    clval = [] #np.expand_dims(frame[pos[1],pos[0]],0)
    while (ret):    
        clval.append(frame[pos[1],pos[0]])
        ret, frame = cap.read()
    
    cap.release()
    print('End reading video')
    
    digi = np.array(clval).sum(1)

    digi = digi>100
    digi = digi.astype(int)
    
    switch = np.diff(digi, axis=0)
    
    print('Start reading .xdf')
    stream = xdf.load_xdf(xdf_file, verbose = False)
    print('End reading .xdf')
    
    stream_names = np.array([item['info']['name'][0] for item in stream[0]])
    vid_sync = list(stream_names).index('Video Sync')
    
    vid_sync_times = np.array(stream[0][vid_sync]['time_stamps'])
    
    get_indices = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    record_frames = np.array(get_indices(1,switch))+1

    
    me,be = np.polyfit(record_frames[-2:], vid_sync_times[-2:], 1)
    ms, bs = np.polyfit(record_frames[:2], vid_sync_times[:2], 1)
    
    all_frames = np.arange(len(clval))
    frame_time = np.interp(all_frames, record_frames,vid_sync_times)
    frame_time[:record_frames[0]]= all_frames[:record_frames[0]]*ms+bs
    frame_time[record_frames[-1]+1:]=all_frames[record_frames[-1]+1:]*me+be
    
    return(frame_time)

#Export data into .fif for visualization in other programs
def fif_save(raw, fif):
    
    evt = mne.Annotations(onset=raw.times[raw.events[:, 0]], duration=np.zeros(len(raw.events)), description=["Event/" + list(raw.event_id.keys())[here] for here in raw.events[:, 2]])
    
    raw.set_annotations(evt)
    
    raw.save(fif)


'''
def fif_loader(fif):
    raw = mne.io.Raw(fif)
    stims = raw.pick_types(meg=False, stim=True)
'''


def quickplot(raw):
    # quick data visualization that filters Oculus specifc noise
    raw.drop_channels(["EOG_U"])
    raw.notch_filter(np.arange(50,2500,50))
    raw.notch_filter(np.arange(90,2500,90))
    raw.notch_filter(np.arange(52.1,2500,52.1))
    raw.set_eeg_reference()
    raw.plot(remove_dc = True, scalings='auto', events=raw.events, event_id=raw.event_id)


def psd_topo(raw):
    montage = mne.channels.read_montage("standard_1005",ch_names=raw.info["ch_names"])
    raw.set_montage(montage)
    raw.plot_psd_topo(fmax=250)


def static_epochs(path, files, regexp, timeframe_start, timeframe_end, target_fps):
    """ Extracts and accumulates given timeframes around events from different .xdf files.
    Args:
        path: If the files share a single path, you can specify it here.
        
        files: A list of .xdf files to extract data from.
        
        regexp: A regular expression that defines the format of the extracted events.
        
        timeframe_start: The time in seconds before the event, in which the EEG Data is extracted.
        
        timeframe_end: The time in seconds after the event, in which the EEG Data is extracted.
        
        target_fps: Downsample the EEG-data to this value.         
    
    Returns:
        epoch: An mne Epochs object
    """
    
    master_id = {}
    epoch = []
    master_legend = []
    for file in files:
        print('Reading ', file)
        current_raw = xdf_loader(path+file)
        current_events = current_raw.events
        current_id = current_raw.event_id
        
        current_raw._data = exponential_running_standardize(current_raw._data, factor_new=0.001, init_block_size=None, eps=0.0001)
        #current_raw.filter(1,40)
        
        # Compute which actions are available in the current file
        here = np.array([bool(re.search(regexp, element)) for element in list(current_id.keys())])
        legend = np.array(list(current_id.keys()))[here]
        
        # Update Master legend and ID if the current file includes new actions
        for event in legend[[item not in master_legend for item in legend]]:
            master_id[event] = len(master_id)
            master_legend = np.append(master_legend, event)

        picked_events = np.empty([0,3], dtype=int)    
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
        current_epoch = mne.Epochs(current_raw, picked_events, picked_id, tmin=-timeframe_start, tmax=timeframe_end)
        current_epoch.load_data()
        current_epoch.resample(target_fps)
        
        
        # Append the current epochs if there are epochs to append to
        if not epoch:
            epoch = current_epoch.copy()
        else:
            epoch = mne.EpochsArray(np.append(epoch[:].get_data(), current_epoch[:].get_data(), axis=0), info=epoch.info, events=np.append(epoch.events, current_epoch.events, axis=0), event_id=master_id)
        
    return epoch

def dlvr_braindecode(path, files, timeframe_start, target_fps, preprocessing=True):
    """ Uses event markers to extract motor tasks from multiple DLVR .xdf files.
    Args:
        path: If the files share a single path, you can specify it here.
        
        files: A list of .xdf files to extract data from.
        
        timeframe_start: The time in seconds before the event, in which the EEG Data is extracted.
        
        target_fps: Downsample the EEG-data to this value.     

        preprocessing: Filters and demean/unitvariance activated (true) or deactivated (false)
    
    Returns:
        X: A list of trials
        y: An array specifying the action
    """
    
    # Epochs list containing differently sized arrays [#eeg_electrodes, times]
    X = []
    #event ids corresponding to the trials where 'left' = 0 and 'right' = 1
    y = np.array([])
    for file in files:
        #load a file
        print('Reading ', file)
        current_raw = xdf_loader(path+file)
        
        # For MEGVR experiments switch EMG into C3/4
        current_raw._data[14,:]= current_raw.get_data(picks = ['EMG_LH']) #C3
        current_raw._data[16,:]= current_raw.get_data(picks = ['EMG_RH']) #C4
        
        #discard EOG/EMG
        current_raw.pick_types(meg=False, eeg=True)
        #pick only relevant events
        events = current_raw.events[(current_raw.events[:,2]==current_raw.event_id['Monster left']) | (current_raw.events[:,2]==current_raw.event_id['Monster right'])]
        #timestamps where a monster deactivates
        stops = current_raw.events[:,0][(current_raw.events[:,2]==current_raw.event_id['Monster destroyed'])]
        #timestamps where trials begin
        starts = events[:,0]
    
        #extract event_ids and shift them to [0, 1]
        key = events[:,2]
        key = (key==key.max()).astype(np.int64)
        
        #standardize, convert to size(time, channels)
        #current_raw._data = exponential_running_standardize(current_raw._data.T, factor_new=0.001, init_block_size=None, eps=0.0001).T
        
        #Find the trials and their corresponding end points
        
        bads = np.array([])
        for count, event in enumerate(starts):
            #in case the last trial has no end (experiment ended before the trial ends), discard it
            if len(stops[stops>event]) == 0:
                key = key[:-sum(starts>=event)]
                break
                
            if stops[stops>event][0]-event < 5000:
                bads = np.append(bads,count)                
                continue
            
            #Get the trial from 1 second before the task starts to the next 'Monster deactived' flag
            current_epoch = current_raw._data[:, event-round(timeframe_start*5000) : stops[stops>event][0]]
            
            if preprocessing == True:
                #filter signal
                B_1, A_1 = butter(5, 1, btype='high', output='ba', fs = 5000)

                # Butter filter (lowpass) for 30 Hz
                B_40, A_40 = butter(6, 120, btype='low', output='ba', fs = 5000)

                # Notch filter with 50 HZ
                F0 = 50.0
                Q = 30.0  # Quality factor
                # Design notch filter
                B_50, A_50 = iirnotch(F0, Q, 5000)
        
                current_epoch = filtfilt(B_50, A_50, current_epoch)
                current_epoch = filtfilt(B_40, A_40, current_epoch)
                #current_epoch = filtfilt(B_1, A_1, current_epoch)
            
            
            #downsample to 250 Hz
            current_epoch= resampy.resample(current_epoch, 5000, 250,axis=1)
            current_epoch = current_epoch.astype(np.float32)
            
            if preprocessing == True:
                #standardize, convert to size(time, channels)
                current_epoch = exponential_running_standardize(current_epoch.T, factor_new=0.001, init_block_size=None, eps=0.0001).T
            
            
            X.append(current_epoch)
        
        if len(bads)>0:
            key = np.delete(key, bads)
        y = np.append(y,key)
    y = y.astype(np.int64)
    return(X,y)


def bdonline_extract(path, files, timeframe_start, target_fps):
    """ Uses event markers to extract motor tasks from multiple DLVR .xdf files.
    Args:
        path: If the files share a single path, you can specify it here.
        
        files: A list of .xdf files to extract data from.
        
        timeframe_start: The time in seconds before the event, in which the EEG Data is extracted.
        
        target_fps: Downsample the EEG-data to this value.         
    
    Returns:
        X: A list of trials
        y: An array specifying the action
    """
    
    DOWNSAMPLING_COEF = int(5000 / target_fps)
    processor = StandardizeProcessor()
    # Epochs list containing differently sized arrays [#eeg_electrodes, times]
    X = []
    #event ids corresponding to the trials where 'left' = 0 and 'right' = 1
    y = np.array([])
    for file in files:
        #load a file
        print('Reading ', file)
        current_raw = xdf_loader(path+file)
        
        # For MEGVR experiments switch EMG into C3/4
        current_raw._data[14,:]= current_raw.get_data(picks = ['EMG_LH']) #C3
        current_raw._data[16,:]= current_raw.get_data(picks = ['EMG_RH']) #C4
        
        #discard EOG/EMG
        current_raw.pick_types(meg=False, eeg=True)
        #pick only relevant events
        events = current_raw.events[(current_raw.events[:,2]==current_raw.event_id['Monster left']) | (current_raw.events[:,2]==current_raw.event_id['Monster right'])]
        #timestamps where a monster deactivates
        stops = current_raw.events[:,0][(current_raw.events[:,2]==current_raw.event_id['Monster destroyed'])]
        #timestamps where trials begin
        starts = events[:,0]
    
        #extract event_ids and shift them to [0, 1]
        key = events[:,2]
        key = (key==key.max()).astype(np.int64)
        
        #standardize, convert to size(time, channels)
        #current_raw._data = exponential_running_standardize(current_raw._data.T, factor_new=0.001, init_block_size=None, eps=0.0001).T
        
        #Find the trials and their corresponding end points
        #filter signal
        B_1, A_1 = butter(5, 1, btype='high', output='ba', fs = 5000)

        # Butter filter (lowpass) for 30 Hz
        B_40, A_40 = butter(6, 120, btype='low', output='ba', fs = 5000)

        # Notch filter with 50 HZ
        F0 = 50.0
        Q = 30.0  # Quality factor
        # Design notch filter
        B_50, A_50 = iirnotch(F0, Q, 5000)
        
        bads = np.array([])
        for count, event in enumerate(starts):
            #in case the last trial has no end (experiment ended before the trial ends), discard it
            if len(stops[stops>event]) == 0:
                key = key[:-sum(starts>=event)]
                break
                
            if stops[stops>event][0]-event < 5000:
                bads = np.append(bads,count)                
                continue
            
            #Get the trial from 1 second before the task starts to the next 'Monster deactived' flag
            current_epoch = np.array([]) #current_raw._data[:, event-round(timeframe_start*5000) : stops[stops>event][0]]
            #400 samples each
            ii = event-round(timeframe_start*5000)
            while ii < stops[stops>event][0]:
                current_chunk = current_raw._data[:,ii:ii+400] 
                ii += 400
        
                current_chunk = filtfilt(B_50, A_50, current_chunk)
                current_chunk = filtfilt(B_40, A_40, current_chunk)            
            
                #downsample to 250 Hz
                current_chunk = np.array([np.mean(current_chunk[:, i:i + DOWNSAMPLING_COEF], axis=1) for i in
                            np.arange(0, 400, DOWNSAMPLING_COEF)]).astype('float32')
                
            
                #standardize, convert to size(time, channels)
                #current_epoch = exponential_running_standardize(current_epoch.T, factor_new=0.001, init_block_size=None, eps=0.0001).T
                current_chunk = processor.process(current_chunk).T
                
                current_chunk = current_chunk.astype(np.float32)
                
                if current_epoch.size == 0:
                    current_epoch = current_chunk                  
                else:
                    current_epoch = np.append(current_epoch, current_chunk, 1)  
                    
            
            X.append(current_epoch)
        
        if len(bads)>0:
            key = np.delete(key, bads)
        y = np.append(y,key)
    y = y.astype(np.int64)
    return(X,y)



def plot_relative_psd(epochs, events, fmax=np.inf):
    epochs.pick_types(meg=False, eeg=True)
    
    cond_1 = epochs[events[0]]
    cond_2 = epochs[events[1]]

    psds_1, freqs = mne.time_frequency.psd_welch(cond_1,n_fft=int(epochs.info['sfreq']), fmax=fmax)
    psds_2,_ = mne.time_frequency.psd_welch(cond_2, n_fft=int(epochs.info['sfreq']), fmax=fmax)

    # psds_1 = 20 * np.log10(psds_1)
    # psds_2 = 20 * np.log10(psds_2)
    
    mean_1 = np.median(psds_1, axis=0)
    mean_2 = np.median(psds_2, axis=0)
    
    div = 20 * np.log10(mean_1/mean_2)
    
    def my_callback(ax, ch_idx):
        """
        This block of code is executed once you click on one of the channel axes
        in the plot. To work with the viz internals, this function should only take
        two parameters, the axis and the channel or data index.
        """
        ax.plot(freqs, div[ch_idx], color='red')
        ax.set_xlabel = 'Frequency (Hz)'
        ax.set_ylabel = 'Power (dB)'

    f = plt.figure()
    for ax, idx in mne.viz.iter_topography(epochs.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                               on_pick=my_callback,
                                           fig=f):
        ax.plot(div[idx], color='red')
        ax.axhline(color='black')
        ax.axvline(color='black')

    plt.gcf().suptitle('Power spectral densities')
    plt.show()


def plot_dual_psd(epochs, events, fmax=np.inf):
    epochs.pick_types(meg=False, eeg=True)

    cond_1 = epochs[events[0]]
    cond_2 = epochs[events[1]]

    psds_1, freqs = mne.time_frequency.psd_welch(cond_1, n_fft=int(epochs.info['sfreq']), fmax=fmax)
    psds_2,_ = mne.time_frequency.psd_welch(cond_2, n_fft=int(epochs.info['sfreq']), fmax=fmax)

    psds_1 = 20 * np.log10(psds_1)
    psds_2 = 20 * np.log10(psds_2)

    mean_1 = np.median(psds_1, axis=0)
    mean_2 = np.median(psds_2, axis=0)

    def my_callback(ax, ch_idx):
        """
        This block of code is executed once you click on one of the channel axes
        in the plot. To work with the viz internals, this function should only take
        two parameters, the axis and the channel or data index.
        """
        ax.plot(freqs, mean_1[ch_idx], color='red')
        ax.plot(freqs, mean_2[ch_idx], color='blue')
        ax.set_xlabel = 'Frequency (Hz)'
        ax.set_ylabel = 'Power (dB)'

    f = plt.figure()
    for ax, idx in mne.viz.iter_topography(epochs.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                               on_pick=my_callback,
                                           fig=f):
        ax.plot(mean_1[idx], color='red')

    for ax, idx in mne.viz.iter_topography(epochs.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                                           fig=f):
        ax.plot(mean_2[idx], color='blue')

    plt.gcf().suptitle('Power spectral densities')
    plt.show()


def plot_relative_tfr(epochs, events, timeframe_start, timeframe_stop):
    #epochs.pick_types(meg=False, eeg=True)
    
    cond_1 = epochs[events[0]]
    cond_2 = epochs[events[1]]
    
    wsize = 120 #int(epochs.info['sfreq'] + epochs.info['sfreq'] % 4)
    tstep = int(wsize/10)
    freqs = mne.time_frequency.stftfreq(wsize=wsize, sfreq=epochs.info['sfreq'])
    
    img_1 = [mne.time_frequency.stft(cond_1._data[x], wsize, tstep) for x in np.arange(cond_1._data.shape[0])]
    img_2 = [mne.time_frequency.stft(cond_2._data[x], wsize, tstep) for x in np.arange(cond_2._data.shape[0])]
    
    data_1 = np.abs(img_1)**2
    data_2 = np.abs(img_2)**2
    
    mean_1 = np.median(data_1, axis=0)
    mean_2 = np.median(data_2, axis=0)
    
    div = mean_1/mean_2
    
    tfr = mne.time_frequency.AverageTFR(info=epochs.info, data=div, times=np.arange(timeframe_start, timeframe_stop, .1), freqs=freqs, nave=len(epochs[events[0]])+len(epochs[events[1]]))
    
    tfr.plot_topo(fmax=55, dB=True, title = ('TFR for "' + events[0] + '"/"' + events[1] + '"'))


def plot_relsw(epochs, events, fmax= np.inf):
    #epochs.pick_types(meg=False, eeg=True)

    cond_1 = epochs[events[0]]
    cond_2 = epochs[events[1]]

    wsize = int(epochs.info['sfreq'] + epochs.info['sfreq'] % 4)
    tstep = int(wsize/10)
    freqs = mne.time_frequency.stftfreq(wsize=wsize, sfreq=epochs.info['sfreq'])

    img_1 = [mne.time_frequency.stft(cond_1._data[x], wsize, tstep) for x in np.arange(cond_1._data.shape[0])]
    img_2 = [mne.time_frequency.stft(cond_2._data[x], wsize, tstep) for x in np.arange(cond_2._data.shape[0])]

    data_1 = np.abs(img_1)**2
    data_2 = np.abs(img_2)**2

    mean_1 = np.median(data_1, axis=3)
    mean_2 = np.median(data_2, axis=3)

    mean_1 = np.median(mean_1, axis=0)
    mean_2 = np.median(mean_2, axis=0)

    div = 20 * np.log10(mean_1/mean_2)

    def my_callback(ax, ch_idx):
        """
        This block of code is executed once you click on one of the channel axes
        in the plot. To work with the viz internals, this function should only take
        two parameters, the axis and the channel or data index.
        """
        ax.plot(freqs[freqs <= fmax], div[ch_idx][freqs <= fmax], color='red')
        ax.set_xlabel = 'Frequency (Hz)'
        ax.set_ylabel = 'Power (dB)'

    f = plt.figure()
    for ax, idx in mne.viz.iter_topography(epochs.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                               on_pick=my_callback,
                                           fig=f):
        ax.plot(freqs[freqs <= fmax], div[idx][freqs <= fmax], color='red')
        ax.axhline(color='black')
        ax.axvline(color='black')

    plt.gcf().suptitle('Power spectral densities')
    plt.show()


def plot_dualsw(epochs, events, fmax=np.inf):
    #epochs.pick_types(meg=False, eeg=True)

    cond_1 = epochs[events[0]]
    cond_2 = epochs[events[1]]

    wsize = int(epochs.info['sfreq']+ epochs.info['sfreq'] % 4)
    tstep = int(wsize/10)
    freqs = mne.time_frequency.stftfreq(wsize=wsize, sfreq=epochs.info['sfreq'])

    img_1 = [mne.time_frequency.stft(cond_1._data[x], wsize, tstep) for x in np.arange(cond_1._data.shape[0])]
    img_2 = [mne.time_frequency.stft(cond_2._data[x], wsize, tstep) for x in np.arange(cond_2._data.shape[0])]

    data_1 = np.abs(img_1)**2
    data_2 = np.abs(img_2)**2

    mean_1 = np.median(data_1, axis=3)
    mean_2 = np.median(data_2, axis=3)

    mean_1 = np.median(mean_1, axis=0)
    mean_2 = np.median(mean_2, axis=0)

    div = 20 * np.log10(mean_1/mean_2)

    def my_callback(ax, ch_idx):
        """
        This block of code is executed once you click on one of the channel axes
        in the plot. To work with the viz internals, this function should only take
        two parameters, the axis and the channel or data index.
        """
        ax.plot(freqs[freqs <= fmax], mean_1[ch_idx][freqs <= fmax], color='red')
        ax.plot(freqs[freqs <= fmax], mean_2[ch_idx][freqs <= fmax], color='blue')
        ax.set_xlabel = 'Frequency (Hz)'
        ax.set_ylabel = 'Power (dB)'

    f = plt.figure()
    for ax, idx in mne.viz.iter_topography(epochs.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                               on_pick=my_callback,
                                           fig=f):
        ax.plot(freqs[freqs <= fmax], mean_1[idx][freqs <= fmax], color='red')

    for ax, idx in mne.viz.iter_topography(epochs.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                                           fig=f):
        ax.plot(freqs[freqs <= fmax], mean_2[idx][freqs <= fmax], color='blue')

    plt.gcf().suptitle('Power spectral densities')
    plt.show()


def plot_tfr_topomap(epochs, events, fmin=0, fmax=np.inf, reject=None):
    epochs.pick_types(meg=False, eeg=True)
    epochs.set_eeg_reference()

    cond_1 = epochs[events[0]]
    cond_2 = epochs[events[1]]

    wsize = 120 #int(epochs.info['sfreq']+ epochs.info['sfreq']%4)
    tstep = 12#int()
    freqs = mne.time_frequency.stftfreq(wsize=wsize, sfreq=epochs.info['sfreq'])

    img_1 = [mne.time_frequency.stft(cond_1._data[x], wsize, tstep) for x in np.arange(cond_1._data.shape[0])]
    img_2 = [mne.time_frequency.stft(cond_2._data[x], wsize, tstep) for x in np.arange(cond_2._data.shape[0])]

    data_1 = np.abs(img_1)**2
    data_2 = np.abs(img_2)**2

    avg_1 = np.median(data_1, axis=3)
    avg_2 = np.median(data_2, axis=3)
    
    if reject:
        split = avg_1.shape[0]
        val = np.append(avg_1, avg_2, axis=0)[:, :, 201:].sum(axis=(1, 2))

        # plt.figure()
        # plt.hist(val, bins=50)
        # plt.show()
        # plt.gcf().suptitle("Trial/Power Histogram")
        # plt.xlabel("Power")
        # plt.ylabel("# Trials")
        
        ind = val.argsort()[-reject:]
        
        avg_1 = np.delete(avg_1, ind[ind<split], axis=0)
        avg_2 = np.delete(avg_2, ind[ind>=split]-split, axis=0)

    mean_1 = np.median(avg_1, axis=0)
    mean_2 = np.median(avg_2, axis=0)

    div = 20 * np.log10(mean_1/mean_2)

    fig, ax = plt.subplots()
    im = mne.viz.plot_topomap(np.median(div[:, (freqs >= fmin) & (freqs <= fmax)], axis=1), pos=epochs.info)
    plt.gcf().suptitle(('Log relative power of \n"' + events[0] + '"/"' + events[1] + '"'))
    plt.colorbar(im[0])

