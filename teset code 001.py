import pandas as pd
import numpy as np
from numpy.fft import fft
from numpy.fft import fftshift

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.signal import spectrogram

############ LOAD DATA

# Replace 'path_to_your_file' with the actual path to your file in OneDrive
file_path = 'C:/Users/Sverrir/OneDrive - Menntaský/FARICE/tmp_FARICE/Fs.npy'
file_path2 = 'C:/Users/Sverrir/OneDrive - Menntaský/FARICE/tmp_FARICE/sig.npy'

# Load the files
data1 = np.load(file_path)
#print(data1.shape)
#print(data1.size)

data = np.load(file_path2)
#print(data.shape)
#print(data.size)

########### Select DataSet and show on graph #########

span = 2
line1 = data[span-1]
fs = data1
print(line1.size)

## np.arange(start, stop, step):
td = np.arange(0, 1/(60*60*fs) * line1.size, 1/(60*60*fs))

#plt.figure(figsize=(12,3))
plt.figure                                  #Fig.1
legText = "wave gögn - span " + str(span)
plt.plot(td,line1-np.mean(line1), label=legText)
plt.xlabel('Time [hours]')
plt.ylabel('Útslag frquency [Hz]?')
plt.legend()
#plt.show()

########### Make a sectrogram of data and show

frequencies, times, Sxx = spectrogram(line1, fs)

# Plot the spectrogram
plt.figure(figsize=(10, 6))                 #Fig.2
plt.subplot(1, 2, 1)  # (rows, cols, index)
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto', cmap='Spectral_r', vmin=0, vmax=50) #shading='gouraud')
#plt.yscale('log')
plt.title("Spectrogram")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
plt.colorbar(label="Power [dB]")
plt.tight_layout()
#plt.show()

############

# Compute the spectrogram
frequencies, times, Sxx = spectrogram(line1, fs*4, nfft=1024*10, nperseg=256*2, noverlap=10)

# frequencies, times, Sxx = spectrogram(
#    x,            # Input signal (1D array)
#    fs=1.0,       # Sampling frequency of the input signal
#    window='hann',# Desired window function (default: 'hann') e.g., 'hann', 'hamming', etc.
#    nperseg=256,  # Length of each segment (default: 256) for FFT
#    noverlap=None,# Number of points to overlap between segments (default: None)
#    nfft=None,    # Number of FFT points (default: None)
#    detrend='constant', # How to detrend each segment (default: 'constant')
#    return_onesided=True, # Return a one-sided spectrogram (default: True)
#    scaling='density',    # Scaling of the spectrum ('density' or 'spectrum')
#    axis=-1        # Axis along which to compute the spectrogram
#)

#-------------------------------

print("Array Sxx.shape:", Sxx.shape)
print("Array frequencies.shape:", frequencies.shape)
print("Array times.shape:", times.shape)

# Plot the spectrogram
#plt.figure(figsize=(10, 6))         #Fig.3
plt.subplot(1, 2, 2)  # (rows, cols, index)
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto', cmap='Spectral_r', vmin=0, vmax=50)  #cmap = viridis, plasma, inferno, magma, coolwarm, Spectral_r, RdYlGn_r
plt.title("Spectrogram")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
#plt.yscale('log')
plt.colorbar(label="Power [dB]")
plt.tight_layout()
#ax = plt.gca()
#ax.yaxis.set_major_locator(LogLocator(base=10.0))
#plt.show()

###################################

# Filter frequencies to range from 10^-2 to 10^2
min_freq, max_freq = 1e-2, 1e2
freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
filtered_frequencies = frequencies[freq_mask]
filtered_Sxx = Sxx[freq_mask, :]  # Select rows corresponding to the filtered frequencies

# Plot the spectrogram with logarithmic y-axis
plt.figure(figsize=(10, 6))                 #Fig.4
plt.pcolormesh(times, filtered_frequencies, 10 * np.log10(filtered_Sxx), shading='gouraud', cmap='Spectral_r', vmin=0, vmax=50)
plt.yscale('log')  # Set y-axis to logarithmic
plt.colorbar(label='Power (dB)')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('Spectrogram (Frequency Range: $10^{-2}$ to $10^{2}$ Hz)')
plt.ylim(min_freq, max_freq)  # Set the y-axis range explicitly
plt.show()


########### Use My own FFT to make spectrogram ?????????????'

#fft_line1 = fft(line1[0:10000]-np.mean(line1[0:10000]))
fft_line1 = fft(line1-np.mean(line1))
fft_line1 = fftshift(fft_line1)
#magnitudeXX = np.abs(XX)
#phaseXX = np.angle(XX)

### pi/12 er einu sinni á 12 mánuðum, því T_0=12
#T0 = 12
#wk=np.linspace(-T0/2,T0/2,len(heild))
sampling_rate = data1
#f = np.fft.fftfreq(len(fft_line1), d=1/sampling_rate)
f=np.linspace(-data1/2,data1/2,len(fft_line1))

#plt.figure(figsize=(12,4))
##plt.plot(abs(fft_line1[100000:150000])) # label=legText)
#plt.plot(f, abs(fft_line1)) # label=legText)
#plt.show()
