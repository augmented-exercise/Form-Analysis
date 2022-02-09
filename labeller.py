#! /usr/bin/env python3

# External imports
import os

# Internal imports
import sheets_connect
import repanalysis

# Helper functions

def convert_to_dict(arr):
    """
    Convert a nested list into a list of dicts. First row is the headers
    """
    vals = []
    headers = arr[0]
    for elem in arr[1:]:
        out = {}
        for i, name in enumerate(headers):
            out[name] = elem[i]
        vals.append(out)
    return vals

# Logical driver functions

def label(filename, tags):
    """
    Label a singel file

    tags: list of table entries containing about the files
    """
    valid = lambda x : x["Time"] in filename and x["Valid"] == "TRUE"
    candidate_rows = list(filter(valid, tags))
    if len(candidate_rows) == 0:
        print(f'No data found for file {filename}')
        return
    if len(candidate_rows) > 1:
        print(f'More than one entry for file {filename}. Taking the first one')
    info = candidate_rows[0]
    
    # Now load the data and produce our pandas data frames
    accel, gyro = repanalysis.import_data(filename)
    repanalysis.preprocess_data(accel, gyro)
    accel = repanalysis.linear_interpolate(accel)
    gyro = repanalysis.linear_interpolate(gyro)

    save_dir = "labeller"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    write_raw = True
    if write_raw:
        base_name, _ = os.path.splitext(filename)
        base_name = os.path.basename(base_name)
        write_name = os.path.join(save_dir, base_name)
        accel.to_csv(write_name+"_accel.csv")
        gyro.to_csv(write_name+"_gyro.csv")

def main():
    """
    Loop over files and label them
    """
    # Load info about files from sheets
    sheet = sheets_connect.connect()
    tags = sheets_connect.get(sheet, "Timestamps!A1:G23")
    tags = convert_to_dict(tags)

    # filename = "data/Wed Feb 02 15:38:27 MST 2022"
    data_folder = "data"
    for file in os.listdir(data_folder):
        label(os.path.join(data_folder, file), tags)

if __name__ == "__main__":
    main()
