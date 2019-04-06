#!/usr/bin/env python
# Code from https://dsp.stackexchange.com/questions/32076/fft-to-spectrum-in-decibel

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
import sys

plt.close('all')


def dbfft(x, fs, win=None, ref=32768):
    """
    Calculate spectrum in dB scale
    Args:
        x: input signal
        fs: sampling frequency
        win: vector containing window samples (same length as x).
             If not provided, then rectangular window is used by default.
        ref: reference value used for dBFS scale. 32768 for int16 and 1 for float

    Returns:
        freq: frequency vector
        s_db: spectrum in dB scale
    """

    N = len(x)  # Length of input sequence

    if win is None:
        win = np.ones(1, N)
    if len(x) != len(win):
            raise ValueError('Signal and window must be of the same length')
    x = x * win

    # Calculate real FFT and frequency vector
    sp = np.fft.rfft(x)
    freq = np.arange((N / 2) + 1) / (float(N) / fs)

    # Scale the magnitude of FFT by window and factor of 2,
    # because we are using half of FFT spectrum.
    s_mag = np.abs(sp) * 2 / np.sum(win)

    # Convert to dBFS
    s_dbfs = 20 * np.log10(s_mag/ref) 

    return freq, s_dbfs

def harmonic_width(freq, s_db, peak_freq, d, threshold):
    peak_freq_i = 0
    for i in range(len(freq)):
        if freq[i] >= peak_freq:
            peak_freq_i = i
            break

    peak_height = s_db[peak_freq_i]
    while (s_db[peak_freq_i + 1] > s_db[peak_freq_i] or peak_height < threshold): # we assume that the given peak_freq always lags behind the true peak freq
        peak_freq_i += 1
        peak_height = s_db[peak_freq_i]

    while (s_db[peak_freq_i - 1] > s_db[peak_freq_i]):
        peak_freq_i -= 1
        peak_height = s_db[peak_freq_i]

    peak_freq = freq[peak_freq_i]
    target_height = peak_height - d

    # Climb left down the peak until you reach target_height or lower
    for i in range(peak_freq_i, 0, -1):
        if s_db[i] == target_height:
            left_point_x = freq[i]
            left_point_y = target_height
            break
        elif s_db[i] < target_height:
            left_point_x = freq[i]
            left_point_y = s_db[i]
            upper_point_x = freq[i + 1]
            upper_point_y = s_db[i + 1]
            slope = (upper_point_y - left_point_y) / (upper_point_x - left_point_x)
            #left_point_x = slope / target_height
            #left_point_y = target_height
            break

    # Now the right side of the peak...
    right_point_x = freq[peak_freq_i]
    for i in range(peak_freq_i, len(freq)):
        if s_db[i] <= target_height:
            right_point_x = freq[i]
            right_point_y = target_height
            break
        elif s_db[i] < target_height:
            right_point_x = freq[i]
            right_point_y = s_db[i]
            upper_point_x = freq[i - 1]
            upper_point_y = s_db[i - 1]
            slope = (right_point_y - upper_point_y) / (right_point_x - upper_point_x)
            #right_point_x = slope / target_height
            #right_point_y = target_height
            break

    return right_point_x - left_point_x, left_point_x, right_point_x, peak_freq


def main():
    # Load the file
    fs, signal = wf.read(sys.argv[1])
    signal = [signal_i[0] for signal_i in signal]

    # Take slice
    N = 32768 # 8192
    win = np.hamming(N)
    freq, s_dbfs = dbfft(signal[0:N], fs, win)

    # Scale from dBFS to dB
    K = 120
    s_db = s_dbfs + K

    # Harmonic Profile -- since we know all samples are an eLow, we already know where all the peaks are!
    NBR_HARMONICS = 10
    FUNDAMENTAL_HARMONIC = 82.0953
    HARMONICS = [FUNDAMENTAL_HARMONIC * (i + 1) for i in range(NBR_HARMONICS)]

    d = 15
    threshold = 60 # in my experience, 60 is a pretty good value here

    # Graph it!
    plt.plot(freq, s_db)

    diff_sum = 0
    for i in range(NBR_HARMONICS):
        diff, left, right, peak_freq = harmonic_width(freq, s_db, HARMONICS[i], d, threshold)
        plt.axvline(x=HARMONICS[i], color="black")
        plt.axvline(x=peak_freq, color="green")
        #plt.axvline(x=left, color="red")
        #plt.axvline(x=right, color="orange")
        #plt.plot(freq, [diff]*len(freq))
        diff_sum += diff

    print(diff_sum)
    
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.show()

if __name__ == "__main__":
    main()
