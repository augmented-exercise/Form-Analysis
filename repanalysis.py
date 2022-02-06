#! /usr/bin/env python3
"""
Functions for data analysis of repetitions
"""
# Imports
import numpy as np
import pandas as pd

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
        accel_data = []
        gyro_data = []
        for line in file:
            vals = line.split(',')
            type = vals[0]
            time = int(vals[1])
            data = [float(_) for _ in vals[2:]]
            # Error detection
            if len(data) != 3:
                raise DataError("Not enough data points")
            # Popula
            if type == 'a':
                accel_data.append({
                    'time' : time,
                    'accel' : np.array(data)
                })
            elif type == 'g':
                gyro_data.append({
                    'time' : time,
                    'gyro' : np.array(data)
                })
            else:
                raise DataError("Invalid line type")
    accel_frame = pd.DataFrame(accel_data)
    gyro_frame = pd.DataFrame(gyro_data)
    return accel_frame, gyro_frame

def linear_interpolate(input, interval=0.01):
    """
    Linearly interpolates the accelerometer and gyroscope data to a regular time interval

    Returns: new pandas data frame
    """
    pass

def graph(data):
    """
    Graph the accelerometer data in the pandas dataframe
    """
    pass

def main():
    """
    Main loop to be executed when we call the script
    """
    file = "data/Wed Feb 02 16:14:33 MST 2022"
    accel, gyro = import_data(file)
    print(accel)

# Main

if __name__ == '__main__':
    main()
