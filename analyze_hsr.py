#!/usr/bin/env python

# Code from https://dsp.stackexchange.com/questions/32076/fft-to-spectrum-in-decibel
# Algorithm designed by Paul Rudmik

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
import sys
import glob, os

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

def find_harmonic(freq, s_db, peak_freq):
    peak_freq_i = 0
    for i in range(len(freq)):
        if freq[i] >= peak_freq:
            peak_freq_i = i
            break

    jump = 5

    peak_height = s_db[peak_freq_i]

    while (True):
        for i in range(1, jump):
            if s_db[peak_freq_i + i] > s_db[peak_freq_i]:
                peak_freq_i += i
                break
        if s_db[peak_freq_i] > peak_height:
            peak_height = s_db[peak_freq_i]
        else:
            break

    while (s_db[peak_freq_i - 1] > s_db[peak_freq_i]):
        peak_freq_i -= 1
        peak_height = s_db[peak_freq_i]

    peak_freq = freq[peak_freq_i]

    return peak_freq, peak_freq_i

def harmonic_power(freq, s_db, peak_freq, b, N):
    true_peak, true_peak_i = find_harmonic(freq, s_db, peak_freq)
    left_freq_i = true_peak_i
    right_freq_i = true_peak_i

    for i in range(true_peak_i, 0, -1):
        if freq[i] <= (true_peak - b/2):
            left_freq_i = i
            left_freq = freq[left_freq_i]
            break

    for i in range(true_peak_i, len(freq)):
        if freq[i] >= (true_peak + b/2):
            right_freq_i = i
            right_freq = freq[right_freq_i]
            break

    abs_X_squared = 0
    for i in range(left_freq_i, right_freq_i):
        abs_X_squared += abs(s_db[i]) ** 2

    power = (1/N) * 2 * abs_X_squared

    return power, true_peak, left_freq, right_freq

def total_power(freq, s_db, N):
    abs_X_squared = 0
    for i in range(len(s_db)):
        abs_X_squared += abs(s_db[i]) ** 2

    power = (1/N) * 2 * abs_X_squared

    return power

def analyze_hsr(filename, b = 4, show_graph = False):
    # Load the file
    fs, signal = wf.read(filename)
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

    # Graph it!
    plt.plot(freq, s_db)

    power_sum = 0
    for i in range(NBR_HARMONICS):
        power, true_peak, left, right = harmonic_power(freq, s_db, HARMONICS[i], b, N)
        plt.axvline(x=left, color="red")
        plt.axvline(x=right, color="orange")
        power_sum += power

    total_power_calc = total_power(freq, s_db, N)
    hsr = power_sum / total_power_calc
    
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    if show_graph: plt.show()

    return hsr

def main():
    quiet = False
    if not quiet: print("Harmonic-to-Signal Ratio")
    os.chdir(sys.argv[1])
    for filename in glob.glob("*.wav"):
        if len(sys.argv) > 2:
            hsr = analyze_hsr(filename, int(sys.argv[2]))
        else:
            hsr = analyze_hsr(filename)
        if not quiet: print(filename)
        print(hsr)

if __name__ == "__main__":
    main()
