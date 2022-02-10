#! /usr/bin/env python3
"""
Script which does cross-correlatoin with a sampmle to detect reps
"""

import scipy.signal
import repanalysis
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def cross_correlate(input, reference):
    """"
    Finds the sliding window cross correlatio between the input and
    reference
    """
    output = np.zeros(input.size - reference.size)
    for i in range(output.size):
        window = input[i:i+reference.size]
        corr = np.dot(window, reference)
        corr /=  np.sqrt(np.dot(reference, reference)) # Scale for reference
        corr /= np.sqrt(np.dot(window, window)) # Scale for size of function
        output[i] = corr
    return output

def check(file, reference, plot=False):
    """
    Compute cross correlatoin for a single file with given reference
    """
    # Read the data in from the raw file
    accel, gyro = repanalysis.import_data(file)
    repanalysis.preprocess_data(accel, gyro)
    accel = repanalysis.linear_interpolate(accel)
    gyro = repanalysis.linear_interpolate(gyro)

    # Read the data from our reference
    reference_accel = pd.read_csv(reference, index_col=0)
    cross_corr = cross_correlate(accel.x, reference_accel.x)

    # Find peaks in cross_corr
    peaks, _ = scipy.signal.find_peaks(cross_corr, distance=100, prominence=0.2)

    # Plotting stuff
    if plot:
        fig, ax1 = plt.subplots()
        ax1.plot(accel.x.to_numpy())
        ax2 = ax1.twinx()
        ax2.plot(cross_corr, color='red')
        fig.tight_layout()
        valid = lambda x : x < accel.x.size and cross_corr[x] > 0.3
        peaks = list(filter(valid, peaks))
        for peak in peaks:
            ax2.axvline(peak, linestyle="--", color='black')
        plt.show()
    
    return cross_corr

def main():
    """
    Main function to run when script is run
    """
    # Dumbbell Bench
    # file = "data/Wed Feb 02 15:38:27 MST 2022"  # db bench
    # # file = "data/Wed Feb 02 15:45:50 MST 2022" # Bench press 2
    # # file = "data/test2" # Tejus 2

    # reference = "crosscorrelation/dumbbell_bp.csv"

    # Lateral raise
    file = "data/Wed Feb 02 17:01:42 MST 2022" # The integral had trouble with this one
    reference = "crosscorrelation/lateral_raise.csv"
    check(file, reference, True)

if __name__ == "__main__":
    main()