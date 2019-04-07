#!/usr/bin/env python

# Code from https://scipy-cookbook.readthedocs.io/items/ApplyFIRFilter.html
# Implementation of algorithm described in https://www.researchgate.net/publication/290179759_Objective_measurement_of_music_quality_using_Inter-Band_Relationship_analysis

from __future__ import print_function, division
import time

import numpy as np
from numpy import convolve as np_convolve
from scipy.signal import fftconvolve, lfilter, firwin
from scipy.signal import convolve as sig_convolve
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
import sys, os, glob
import math

from scipy import signal

def analyze_ibr(filename, order = 10, show_grid = False):
    fs, original_data = wf.read(filename)
    IBR_sum = 0
    NBR_TRACKS = len(original_data[0])
    for track_i in range(NBR_TRACKS):
        data = [data_i[track_i] for data_i in original_data]

        # Create Filters as described in paper
        numtaps = order + 1

        f_low = 947 / fs
        f_high = 3186 / fs

        l = firwin(numtaps, f_low)
        b = firwin(numtaps, [f_low, f_high], pass_zero=False)
        h = firwin(numtaps, f_high, pass_zero = False)

        # Apply filters
        l_out = signal.lfilter(l, [1.0], data)
        b_out = signal.lfilter(b, [1.0], data)
        h_out = signal.lfilter(h, [1.0], data)

        # Plot 
        t = np.linspace(0, 1, fs)

        plt.figure
        plt.plot(t, data[:fs], 'b')
        plt.plot(t, l_out[:fs], 'r--')
        plt.plot(t, b_out[:fs], 'r')
        plt.plot(t, h_out[:fs], 'k')
        plt.legend(('initial signal', 'lowpass filter', 'bandpass filter', 'highpass filter'), loc='best')
        plt.grid(True)
        # Enable if you want a preview of the filtered signals before IBR is conducted
        if show_grid: plt.show()

        # Apply IBR Formula as described in paper
        bands = [l_out, b_out, h_out]
        nbr_bands = len(bands)
        drs = [0] * nbr_bands

        # Threshold reported in paper is 4 - we'll use this as an arbitrary sample
        threshold = 4

        for i in range(len(bands)):
            filtered_signal = bands[i]
            x_avg = np.mean(filtered_signal)

            srms_sum = 0
            for x_i in filtered_signal:
                srms_sum += (x_i - x_avg) ** 2

            Spk = max(filtered_signal)

            Srms = math.sqrt((1/len(filtered_signal)) * srms_sum)

            Dr = 20 * math.log10(Spk / Srms)
            drs[i] = Dr

        dr_avg = np.mean(drs)

        drs_sum = 0
        for dr_i in drs:
            drs_sum = (dr_i - dr_avg) ** 2

        IBR = math.sqrt((1 / (nbr_bands - 1)) * drs_sum)

        IBR_sum += IBR
    return IBR_sum / NBR_TRACKS

def main():
    quiet = False
    if not quiet: print("Inter-Band Relationship")
    os.chdir(sys.argv[1])
    for filename in glob.glob("*.wav"):
        if len(sys.argv) > 2:
            ibr = analyze_ibr(filename, int(sys.argv[2]))
        else:
            ibr = analyze_ibr(filename)
        if not quiet: print(filename)
        print(ibr)

if __name__ == "__main__":
    main()
