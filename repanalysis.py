#! /usr/bin/env python3
"""
Functions for data analysis of repetitions
"""
# Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.signal
import scipy.fft
import math
import statistics
import os

PLOTTING = True

# Error
class DataError(Exception):
    """
    Data file had unexpected format
    """

# Kernels
# Modified from https://stackoverflow.com/questions/29920114/how-to-gauss-filter-blur-a-floating-point-numpy-array
def gauss(n=11,sigma=1):
    """
    Generates a guassian kernel for purposes such as smoothing a dataset
    """
    r = range(-int(n/2),int(n/2)+1)
    vals = [1 / (sigma * math.sqrt(2*math.pi)) * math.exp(-float(x)**2/(2*sigma**2)) for x in r]
    vals = np.array(vals)
    return vals/sum(vals) # Normalize
# End citation

# Debugging Functions

def demo_fft(data, label="FFT"):
    fft = scipy.fft.fft(data)
    plt.figure(label)
    plt.plot(fft)

def graph(data):
    """
    Graph the accelerometer data in the pandas dataframe
    """
    if PLOTTING:
        plt.figure()
        plt.plot(data.index, data.x)
        plt.plot(data.index, data.y)
        plt.plot(data.index, data.z)

# Functions

def import_data(file_name : str):
    """
    Imports data from the given file

    Returns: two data frames, one for accelerometer data and one for gyroscope data
    """
    with open(file_name) as file:
        accel_indices = []
        gyro_indices = []
        accel_data = []
        gyro_data = []
        for line in file:
            vals = line.split(',')
            data_type = vals[0]
            time = int(vals[1])
            data = [float(_) for _ in vals[2:]]
            # Error detection
            if len(data) != 3:
                raise DataError("Not enough data points")
            # Popula
            if data_type == 'a':
                accel_indices.append(time)
                accel_data.append({
                    'x' : data[0],
                    'y' : data[1],
                    'z' : data[2],
                })
            elif data_type == 'g':
                gyro_indices.append(time)
                gyro_data.append({
                    'x' : data[0],
                    'y' : data[1],
                    'z' : data[2],
                })
            else:
                raise DataError("Invalid line type")
    accel_indices = pd.TimedeltaIndex(accel_indices)
    gyro_indices = pd.TimedeltaIndex(gyro_indices)
    accel_frame = pd.DataFrame(accel_data, index=accel_indices)
    gyro_frame = pd.DataFrame(gyro_data, index=gyro_indices)
    accel_frame.index.name = 'time'
    gyro_frame.index.name = 'time'
    return accel_frame, gyro_frame

def preprocess_data(accel : pd.DataFrame, gyro : pd.DataFrame):
    """
    Sort by ascending order of timestamp and subtracts the start time from each time stamp.
    Performs this in place.
    """
    # accel.sort_values('time')
    # gyro.sort_values('time')
    start_time = min(accel.index.min(), gyro.index.min())
    accel.index -= start_time
    gyro.index -= start_time

def linear_interpolate(data : pd.DataFrame, interval : int = int(1e7)):
    """
    Linearly interpolates the accelerometer and gyroscope data to a regular time interval

    interval : interval to interpolate to in nanoseconds

    Returns: new pandas data frame
    """
    resampled = data.resample(pd.Timedelta(interval)).mean()
    resampled = resampled.interpolate(method='linear')
    return resampled

# Analysis functions

def get_rms(arr, window_size):
    """
    Efficiently gets the rms average for a sliding window over arr
    """
    square_arr = arr**2
    summed = np.cumsum(square_arr) # Get our cumulative sums
    target_arr = np.zeros(arr.size - window_size)
    # Could theoretically run into problems if our array gets super large but
    # this should be good enough
    for i, _ in enumerate(target_arr):
        target_arr[i] = math.sqrt(summed[i+window_size] - summed[i])
    return target_arr

def get_repetitive(accel):
    """
    Gets repetitve sections of data from a
    """
    window_size = 500 # 5 seconds
    max_freqs = []
    rms = get_rms(accel, window_size)
    for i in range(len(accel)-window_size):
        accel_fft = scipy.fft.fft(accel[i:i+window_size])
        # This rms calculated to be linear in array size
        max_freq = max(np.abs(accel_fft)) # Ignore constant
        max_freqs.append(max_freq/rms[i])
    if PLOTTING:
        plt.figure("Max freqs")
        plt.plot(max_freqs)
        # plt.plot(normalized_max_freqs)
        # plt.plot(accel/max(accel))

    def get_sections(data, threshold=0.90):
        """
        Return list of sections where the values is above a threshold
        """
        going = False
        sections = []
        for i, val in enumerate(data):
            if abs(val) > threshold and not going:
                going = True
                start = i
            elif abs(val) <= threshold and going:
                going = False
                end = i
                sections.append((start,end,))
        if going:
            sections.append((start, len(data)))
        return sections

    def max_section(sections):
        """
        Get the largest section out of a list of sections
        """
        max = 0
        val = None
        for section in sections:
            start, end = section
            size = end-start
            if size > max:
                size = max
                val = section
        return val

    sections = get_sections(max_freqs/max(max_freqs))
    start, end = max_section(sections)
    exercise_section = accel[start:end+window_size]
    exercise_fft = np.abs(scipy.fft.fft(exercise_section))

    if PLOTTING:
        plt.figure("FFT")
        plt.plot(exercise_fft[:100])
    
    # Now find the fundamental frequency of the repetitive section
    return exercise_section, start, end

def integrate(accel, reverse=False, figname=None):
    """
    Attempt to identify start and end of repetitions by taking a double
    integral.

    accel: repetitive portion of the exercise
    """
    accel -= accel.mean()
    vel = np.cumsum(accel)
    vel -= vel.mean()
    pos  = np.cumsum(vel)
    # Subtract out the smoothed version of x_pos to reveal just the details
    n = 150 # Parameter
    pos_smooth = np.convolve(pos, gauss(2*n+1,n))
    diff = pos - pos_smooth[n:-n]
    m = 10 # Parameter
    smoothed_diff = np.convolve(diff, gauss(2*m+1, m))[m:-m]
    # Average
    rms_average = math.sqrt(statistics.mean([x*x for x in smoothed_diff])) # rms
    # Plot
    plt.figure("Integrate")
    plt.clf()
    # plt.plot(accel.index, x_pos)
    # plt.plot(accel.index, x_pos_smooth[5:-5])
    plt.plot(smoothed_diff/smoothed_diff.max())
    plt.plot(accel/accel.max())
    direction = -1 if reverse else 1
    peaks, _ = scipy.signal.find_peaks(direction*smoothed_diff, distance=n,
                                       prominence=[rms_average/2])
    for peak in peaks:
        if peak < len(accel):
            plt.axvline(peak, linestyle='--', color='r')
    # Return the start and end ofthe reps
    if figname:
        plt.savefig(figname)
    return peaks

def lti(accel):
    """
    Tries to use a band pass filter and double integral to get position for the 
    dumbbell bench press.
    """
    interval = 0.01
    lowfreq = 2*math.pi*1/math.sqrt(10) # Avoid accel drift
    highfreq = 1000*2*math.pi # Elminate noise
    # n = 10
    # accel = np.convolve(accel, gauss(2*n+1, n))
    # num, dem = scipy.signal.butter(4, [lowfreq, highfreq], 'bandpass', interval)
    num, dem = scipy.signal.butter(4, lowfreq, 'highpass', interval)
    bandpass = scipy.signal.lti(num, dem)
    times = np.linspace(0, interval*len(accel), len(accel))
    t, out, _ = bandpass.output(accel, times)
    integral = np.cumsum(out)
    t2, out2, _ = bandpass.output(integral, times)
    plt.figure("LTI response")
    # plt.plot(out)
    plt.plot(accel)
    plt.plot(out)
    # plt.plot(out2)

def peak_tracker(accel, reverse=False, figname=None):
    "Attempt to identify start and end of repetitions using peak tracking"
    n = 150
    accel -= accel.mean()
    rms_average = np.sqrt(np.mean(accel**2))
    print(rms_average)
    peaks, _ = scipy.signal.find_peaks(accel, distance=n, prominence=[rms_average])

    plt.figure("Peak track")
    plt.clf()
    # plt.plot(accel.index, x_pos)
    # plt.plot(accel.index, x_pos_smooth[5:-5])
    plt.plot(accel/accel.max())
    for peak in peaks:
        if peak < len(accel):
            plt.axvline(peak, linestyle='--', color='r')
        # Return the start and end ofthe reps
    if figname:
        plt.savefig(figname)
    return peaks

# Write functions

def write(exercise_name : str, form_name : str, data, peaks, label):
    """
    Separates the exercise into repetitions, discarding the first and last rep,
    and then writes the resulting sections to a pandas dataframe in the
    appropriate directory.

    data : pandas data frame containing the resampled and interpolated data
    peaks : list of peaks
    label : e.g. accel or gyro
    """
    # Get write dir name
    root_dir = "repdata"
    write_dir = os.path.join(root_dir,exercise_name,form_name)
    write_dir = write_dir.replace(' ','_') # Make things nicer on our operating systems
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)
    
    # Get reps and write them
    start = peaks[1] # Skip first element [0]
    for i, end_index in enumerate(range(2, len(peaks)-1)): # Ignore last rep
        end = peaks[end_index]
        rep_data = data[start:end]
        rep_data.index -= rep_data.index[0] # Subtract start time
        filename = os.path.join(write_dir, f'{label}-{i}.csv')
        rep_data.to_csv(filename)
        start = end # For next iteration

# Driver routines

def divide(filename, exercise_name, form_name, subject=None, reverse=False):
    """
    Divides the exercise in the file into repetitions and writes the repetitions
    to the file system as pandas csv's
    """
    accel, gyro = import_data(filename)
    preprocess_data(accel, gyro)
    accel_interp = linear_interpolate(accel)
    gyro_interp = linear_interpolate(gyro)

    # Convert to numpy since it's easier to work with
    # In future we should find the best axis
    accel_x_vec = accel_interp.x.to_numpy()
    # demo_fft(accel_x_vec, "Whole fft") # Debugging
    lti(accel_x_vec)
    if PLOTTING:
        plt.figure("Input data")
        plt.plot(accel_x_vec-9.81)
    # Attempt 1 - get repetitive section and then integrate
    reps, start, _ = get_repetitive(accel_x_vec)
    figname = os.path.join("images", f'{exercise_name}-{form_name}')
    if subject:
        figname += f'-{subject}'
    peaks = integrate(reps, reverse=reverse, figname=figname)
    # peaks = peak_tracker(reps, reverse=reverse, figname=figname)
    peaks += start # Account for offset to start of exercise
    extra_label = ""
    if subject:
        extra_label += f"-{subject}"
    if len(peaks > 1): 
        write(exercise_name, form_name, accel_interp, peaks, f"accel{extra_label}")
        write(exercise_name, form_name, gyro_interp, peaks, f"gyro{extra_label}")
    if not PLOTTING:
        plt.close()

def main():
    """
    Main script for testing
    """
    file = None
    reverse = False
    # file, reverse = "data/Wed Feb 02 16:14:33 MST 2022", True # Dumbell curl
    # file, reverse = "data/SP-Timing-Test", True # Shoulder press
    file, reverse = "data/Wed Feb 02 15:38:27 MST 2022", False # db bench
    # file, reverse = "data/Wed Feb 02 15:45:50 MST 2022", False # Bench press 2
    # file, reverse = "data/test2", False # Tejus 2
    # file, reverse = "data/Wed Feb 02 16:32:47 MST 2022", False # Should press wide - Zach
    # file, reverse = "data/Wed Feb 02 17:10:29 MST 2022", False # Tricep pushdown - Zach
    file, reverse = "data/Wed Feb 02 17:13:30 MST 2022", False # Tricep pushdown - Zach
    # file, reverse = "data/Wed Feb 02 16:59:42 MST 2022", False # Lateral raise - Zach
    # file, reverse = "data/Wed Feb 02 16:14:33 MST 2022", False # Curl - Tejus

    divide(file, "test", "test", reverse)
    # Show any plots made during any of the called functions
    if PLOTTING:
        plt.show()
        plt.close()

# Main

if __name__ == '__main__':
    PLOTTING = True
    main()
