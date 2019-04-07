#!/usr/bin/env python

# Code from https://dsp.stackexchange.com/questions/32076/fft-to-spectrum-in-decibel
# Algorithm designed by Paul Rudmik

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
import sys, os, glob

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

def midpoint(x1, y1, x2, y2, target_y):
    assert x1 < x2

    slope = (y2 - y1) / (x2 - x1)
    b = y2 - (slope * x2)
    mid_x = (target_y - b) / slope
    mid_y = target_y

    return mid_x, mid_y

def harmonic_width(freq, s_db, peak_freq, d):
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
    target_height = peak_height - d

    # Climb left down the peak until you reach target_height or lower
    for i in range(peak_freq_i, 0, -1):
        if s_db[i] == target_height:
            left_point_x = freq[i]
            left_point_y = target_height
            break
        elif s_db[i] < target_height:
            left_point_x, left_point_y = midpoint(freq[i], s_db[i], freq[i + 1], s_db[i + 1], target_height)
            break

    # Now the right side of the peak...
    right_point_x = freq[peak_freq_i]
    for i in range(peak_freq_i, len(freq)):
        if s_db[i] == target_height:
            right_point_x = freq[i]
            right_point_y = target_height
            break
        elif s_db[i] < target_height:
            right_point_x, right_point_y = midpoint(freq[i - 1], s_db[i - 1], freq[i], s_db[i], target_height)
            break

    return right_point_x - left_point_x, left_point_x, right_point_x, peak_freq

def analyze_hbm(filename, d = 15, show_graph = False):
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

    diff_sum = 0
    for i in range(NBR_HARMONICS):
        diff, left, right, peak_freq = harmonic_width(freq, s_db, HARMONICS[i], d)
        #plt.axvline(x=HARMONICS[i], color="black")
        #plt.axvline(x=peak_freq, color="green")
        plt.axvline(x=left, color="red")
        plt.axvline(x=right, color="orange")
        diff_sum += diff

    hbm = diff_sum / NBR_HARMONICS
    
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    if show_graph: plt.show()

    return hbm

def main():
    quiet = False
    if not quiet: print("Harmonic Bandwidth Measurement")
    os.chdir(sys.argv[1])
    for filename in glob.glob("*.wav"):
        if len(sys.argv) > 2:
            hbm = analyze_hbm(filename, int(sys.argv[2]))
        else:
            hbm = analyze_hbm(filename)
        if not quiet: print(filename)
        print(hbm)

if __name__ == "__main__":
    main()
