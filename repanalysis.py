#! /usr/bin/env python3
"""
Functions for data analysis of repetitions
"""
# Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.signal

# Error
class DataError(Exception):
    """
    Data file had unexpected format
    """

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

def graph(data):
    """
    Graph the accelerometer data in the pandas dataframe
    """
    plt.figure()
    plt.plot(data.index, data.x)
    plt.plot(data.index, data.y)
    plt.plot(data.index, data.z)

# Analysis functions

def integrate(accel):
    """
    Attempt to identify repetitions by taking a double integral.
    Suffers when there are non-periodic sections of the data
    """
    # Conver to numpy since it's better for data analysis
    times = accel.index.to_numpy(dtype=int)
    x_accel = accel.x.to_numpy()
    # TODO - clean up this code
    x_accel -= x_accel.mean()
    x_vel = np.cumsum(x_accel)
    x_vel -= x_vel.mean()
    x_pos  = np.cumsum(x_vel)
    x_pos /= x_pos.max()
    # Plot
    plt.figure()
    plt.plot(accel.index, x_pos)
    plt.plot(accel.index, accel.x)
    peaks, _ = scipy.signal.find_peaks(x_pos)
    plt.plot([times[peak] for peak in peaks], [x_pos[peak] for peak in peaks], 'ro')
    for peak in peaks:
        plt.axvline(x=times[peak], linestyle='--', color='r')

def main():
    """
    Main loop to be executed when we call the script
    """
    # file = "data/Wed Feb 02 16:14:33 MST 2022" # Dumbell curl
    # file = "SP-Timing-Test" # Shoulder press
    file = "data/Wed Feb 02 15:38:27 MST 2022" # db bench
    accel, gyro = import_data(file)
    preprocess_data(accel, gyro)
    accel_interp = linear_interpolate(accel)
    gyro_interp = linear_interpolate(gyro)
    # graph(accel_interp)
    # graph(gyro_interp)
    integrate(accel)
    plt.show()

# Main

if __name__ == '__main__':
    main()
