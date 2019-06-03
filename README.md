# xdf_mne_interface

## Dependencies
numpy, xdf, matplotlib, re, resampy, mne, braindecode

## TODOs: 
- Implement quick data analysis pipeline:
  1. Load all runs and have a mechanism to keep them separate (e.g. keep them in a list)
  2. Create a downsampled (250 Hz) copy of the data using resampy. USe this copy for all next steps but the spectrum plots
  3. Make a comb filter to remove 50 Hz and 90 Hz and harmonics to make visual inspection possible
  4. Remove DC offset (subtract mean), keeping offset values somewhere
  5. Calls mne raw data viewer on the concatenated runs to perform visual inspection
  6. Calculate and topographically plot power spectrum for each electrode and each file
  7. Calculate and topographically plot variance of each electrode and each file
  8. Epoch the data based on user specified markers and time-windows
  9. Average the epochs for specified markers individually and plot all the marker specific lines with deviation shades for each electrode in a topographical arangement **with axes and legend**
  10. Calculate and topographically plot the average power spectrum for each electrode and each file with one line per marker
  11. Calculate time-frequency spectrograms for each epoch
  12. Calculate baseline (-500 ms - 0 ms before marker, median across time & trials in two steps, all markers pooled)
  13. Divide spectrograms by baseline and take natural log
  14. Plot median spectrogram topographically for each marker
