# xdf_mne_interface
TODOs: 
- Implement quick data analysis pipeline:
  1. Load all runs and have a mechanism to keep them separate (e.g. keep them in a list)
  2. Make a comb filter to remove 50 Hz and 90 Hz and harmonics to make visual inspection possible
  3. Remove DC offset (subtract mean), keeping offset values somewhere
  3. Calls mne raw data viewer on the concatenated runs to perform visual inspection
  4. Calculate and topographically plot power spectrum for each electrode and each file
  5. Calculate and topographically plot variance of each electrode and each file
  6. Epoch the data based on user specified markers and time-windows
  7. Average the epochs for specified markers individually and plot all the marker specific lines with deviation shades for each electrode in a topographical arangement **with axes and legend**
  8. Calculate and topographically plot the average power spectrum for each electrode and each file with one line per marker
  9. Calculate time-frequency spectrograms for each epoch
  10. Calculate baseline (-500 ms - 0 ms before marker, median across time & trials in two steps, all markers pooled)
  11. Divide spectrograms by baseline and take natural log
  12. Plot median spectrogram topographically for each marker
