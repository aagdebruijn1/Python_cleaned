# Import packages needed for these functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
from scipy.signal import correlate, correlation_lags
from scipy.signal import convolve, resample
import json
from scipy.spatial.transform import Slerp
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis, normaltest
import pingouin as pg
from typing import Literal
from scipy.interpolate import interp1d

# Datasets that miss data, contain flipped data or should be time aligned with method 2

# Missing data
non_valid_combinations = {
    ('Lotte', 'straightpush', 'ls'),
    ('Lotte', 'intermittentsprint', 'hs'),
    ('Iza', 'slalom', 'ls'),
    ('Boris', 'pivot', 'hs'), 
    ('Boris', 'straightpush', 'hs'), 
    ("Floris", 'trunkmovement', 'ls'),
    ("Floris", 'trunkmovement', 'hs'),
    ('Hidde', 'straightpush', 'ls'),
    ('Hidde', 'slalom', 'hs'),                  # maybe flipped
    ('Hidde', 'figure8', 'hs'),                 # maybe flipped
    ('Fee', 'straightpush', 'hs'),
    ('Fee','slalom', 'ls'),                     # maybe flipped
    ('Fee', 'pivot', 'hs'),                     # maybe flipped
    ('Fee', 'figure8', 'ls'),
    ('Yinhu', 'slalom', 'ls'),
    ('Lotte', 'intermittentsprint', 'ls'),
    ('Lotte', 'slalom', 'ls')
}

# Noisy low_back 3D MOCAP data
bad_LB_combinations = {
    ('Yinhu', 'figure8', 'ls', 'low_back'),
    ('Yinhu', 'figure8', 'hs', 'low_back'),
    ('Vigo', 'intermittentprint', 'hs', 'low_back'),        # LB = really bad
    ('Vigo', 'uturn', 'ls', 'low_back'),                    # LB = really bad
    ('Yinhu', 'slalom', 'hs', 'low_back')
}

# Data where trunk tilt angles are flipped
flipped_combinations = {
    ('Danique', 'figure8', 'ls', 'low_back'), 
    ('Danique', 'slalom', 'ls', 'low_back'),
    ('Danique', 'uturn', 'ls', 'low_back'),
    ('Holly', 'slalom', 'ls', 'low_back'),
    ('Holly', 'figure8', 'hs', 'low_back'),
    ('Jeske', 'straightpush', 'ls', 'back'),
    ('Jeske', 'uturn', 'ls', 'back'),
    ('Iza', 'slalom', 'hs', 'low_back'),
    ('Iza', 'uturn', 'ls', 'low_back'),
    ('Jeske', 'trunkmovement', 'ls', 'back'), 
    ('Jeske', 'trunkmovement', 'hs', 'back'),
    ('Jeske', 'uturn', 'hs', 'back'), 
    ('Rienk', 'figure8', 'ls', 'low_back'), 
    ('Rienk', 'slalom', 'ls', 'low_back'),
    ('Boris', 'figure8', 'ls', 'low_back'),
    ('Boris', 'slalom', 'ls', 'low_back'), 
    ('Boris', 'slalom', 'hs', 'low_back'),
    ('Fee', 'figure8', 'hs', 'low_back'),
    ('Fee', 'slalom', 'ls', 'back'),
    ('Floris', 'figure8', 'ls', 'low_back'),
    ('Floris', 'slalom', 'ls', 'low_back'),
    ('Hidde', 'figure8', 'ls', 'low_back'),
    ('Hidde', 'figure8', 'hs', 'low_back'),
    ('Hidde', 'slalom', 'ls', 'low_back'),
    ('Hidde', 'slalom', 'hs', 'back'),
    ('Hidde', 'slalom', 'hs', 'low_back'),
    ('Rinske', 'figure8', 'ls', 'low_back'),
    ('Rinske', 'figure8', 'hs', 'low_back'),
    ('Rinske', 'slalom', 'ls', 'low_back'),
    ('Rinske', 'slalom', 'hs', 'low_back'),
    ('Vigo', 'figure8', 'ls', 'low_back'),
    ('Vigo', 'figure8', 'hs', 'low_back'),
    ('Vigo', 'slalom', 'ls', 'low_back'),
    ('Vigo', 'trunkmovement', 'hs', 'back'),
    ('Yinhu', 'figure8', 'ls', 'low_back'),
    ('Yinhu', 'figure8', 'hs', 'low_back'),
    ('Yinhu', 'slalom', 'hs', 'low_back'),
    ('Fee', 'trunkmovement', 'ls', 'back'),
    ('Fee', 'trunkmovement', 'hs', 'back'),
    ('Vigo', 'trunkmovement', 'hs', 'low_back')
}

# Data where the trunk tilt angles include an offset
offset_combinations = {
    ('Danique', 'figure8', 'ls', 'chair'),
    ('Jeske', 'figure8', 'ls', 'chair'),
    ('Boris', 'figure8', 'ls', 'chair'),
    ('Boris', 'figure8', 'hs', 'chair'),
    ('Boris', 'slalom', 'ls', 'chair'),
    ('Boris', 'slalom', 'hs', 'chair'),
    ('Floris', 'figure8', 'ls', 'chair'),
    ('Floris', 'figure8', 'hs', 'chair'),
    ('Floris', 'slalom', 'ls', 'chair'),
    ('Floris', 'slalom', 'hs', 'chair'),
    ('Hidde', 'figure8', 'ls', 'chair'),
    ('Hidde', 'figure8', 'hs', 'chair'),
    ('Hidde', 'pivot', 'hs', 'chair'),
    ('Hidde', 'slalom', 'ls', 'chair'),
    ('Hidde', 'slalom', 'hs', 'chair'),
    ('Rinske', 'figure8', 'ls', 'chair'),
    ('Rinske', 'figure8', 'hs', 'chair'),
    ('Rinske', 'slalom', 'ls', 'chair'),
    ('Rinske', 'slalom', 'hs', 'chair'),
    ('Vigo', 'figure8', 'ls', 'chair'),
    ('Vigo', 'figure8', 'hs', 'chair'),
    ('Vigo', 'slalom', 'ls', 'chair'),
    ('Vigo', 'slalom', 'hs', 'chair'),
    ('Yinhu', 'figure8', 'ls', 'chair'),
    ('Yinhu', 'figure8', 'hs', 'chair'),
    ('Yinhu', 'slalom', 'hs', 'chair')
}

# Time align method: only 360 deg turn included in cross-correlation time alignment
part_combinations = {
    ('Iza', 'starmove', 'ls'),
    ('Lotte', 'intermittentsprint', 'ls'),
    ('Lotte', 'slalom', 'ls'),
    ('Vigo', 'intermittentsprint', 'ls'),
    ('Vigo', 'uturn', 'ls'),
    ('Yinhu', 'intermittentsprint', 'hs'),
    ('Yinhu', 'starmove', 'ls'),
    ('Yinhu', 'starmove', 'hs')
}

# Time align method: use trunk tilt angle for cross-correlation time alignment
ta_combinations = {
    ('Lotte', 'figure8', 'hs'),   
    ('Lotte', 'trunkmovement', 'hs'), 
    ('Boris', 'figure8', 'ls'), 
    ('Boris', 'figure8', 'hs'),
    ('Boris', 'slalom', 'hs'),
    ('Boris', 'starmove', 'ls'), 
    ('Boris', 'starmove', 'hs'), 
    ('Fee', 'intermittentsprint', 'hs'),
    ('Fee', 'slalom', 'hs'),
    ('Floris', 'figure8', 'ls'),
    ('Floris', 'figure8', 'hs'),
    ('Floris', 'intermittentsprint', 'hs'),
    ('Floris', 'pivot', 'hs'),
    ('Floris', 'slalom', 'ls'),
    ('Floris', 'slalom', 'hs'),
    ('Floris', 'uturn', 'ls'),
    ('Floris', 'uturn', 'hs'),
    ('Hidde', 'figure8', 'ls'),
    ('Hidde', 'intermittentsprint', 'ls'),
    ('Hidde', 'pivot', 'hs'),
    ('Hidde', 'slalom', 'ls'),
    ('Hidde', 'starmove', 'ls'),
    ('Hidde', 'starmove', 'hs'),
    ('Hidde', 'trunkmovement', 'ls'),
    ('Hidde', 'trunkmovement', 'hs'),
    ('Rinske', 'figure8', 'ls'),
    ('Rinske', 'figure8', 'hs'),
    ('Rinske', 'pivot', 'ls'),
    ('Rinske', 'pivot', 'hs'),
    ('Rinske', 'slalom', 'ls'),
    ('Rinske', 'slalom', 'hs'),
    ('Rinske', 'starmove', 'ls'),
    ('Rinske', 'starmove', 'hs'),
    ('Rinske', 'trunkmovement', 'ls'),
    ('Rinske', 'trunkmovement', 'hs'),
    ('Vigo', 'figure8', 'ls'),
    ('Vigo', 'figure8', 'hs'),
    ('Vigo', 'pivot', 'hs'),
    ('Vigo', 'slalom', 'ls'),
    ('Vigo', 'slalom', 'hs'),
    ('Vigo', 'starmove', 'ls'),
    ('Vigo', 'starmove', 'hs'),
    ('Vigo', 'trunkmovement', 'ls'),
    ('Yinhu', 'figure8', 'hs'),
    ('Yinhu', 'pivot', 'hs'),
    ('Yinhu', 'slalom', 'hs'),
    ('Yinhu', 'trunkmovement', 'ls'),
    ('Yinhu', 'trunkmovement', 'hs')
}

# Time align method: reinforce a fixed time shift (lag) to align the data in time
fixed_combinations = {
    ('Boris', 'slalom', 'ls'),                  # ts = 252
    ('Fee', 'intermittentsprint', 'ls'),        # ts = 370
    ('Yinhu', 'figure8', 'ls'),                 # ts = 175
    ('Iza', 'starmove', 'ls')                   # ts = 307
}
fixed_time_shift = {'Boris': 252, 'Fee': 370, 'Yinhu': 175, 'Iza': 307}


# ------------------------------------------------------------------- LOAD IMU (RAW GYRO AND MAG) AND 3D MOCAP DATA -------------------------------------------------------------------


def load_imu_data(folder):    
    """
    Loads IMU (Inertial Measurement Unit) data from a given folder containing subfolders for each sensor.

        Parameters:
        folder (str): Path to the main directory containing subdirectories for different IMU sensors.

        Returns:
        tuple:
            - imu_data (dict): A dictionary containing pandas DataFrames of quaternion and rotation matrix data for each IMU sensor.
                Each key is a sensor name (e.g., 'low_back', 'chair', 'l_arm', etc.), and the corresponding value is a DataFrame with:
                    - 'Timestamp (us)': Timestamps in microseconds.
                    - 'X', 'Y', 'Z', 'W': Quaternion components.
                    - 'rot_matrix': Rotation matrices derived from quaternions (np.ndarray of shape (3,3)).
            - imu_metadata (dict): A dictionary containing metadata related to the IMU data:
                - 'imu_names': List of sensor names.
                - 'date_time': List of timestamp metadata extracted from JSON files.
                - 'total_frames': List of total data frames for each sensor.
                - 'rotation_type': Type of rotation representation used ('Quaternion').
                - 'coordinate_space': Reference coordinate space ('Global').
                - 'sampling_freq': Dictionary mapping each sensor name to its computed sampling frequency (Hz).

        Steps:
        1. Identify subdirectories in the provided folder representing individual IMU sensors.
        2. Normalize sensor names (e.g., convert 'chest' to 'low_back').
        3. For each sensor:
            - Load 'Quaternion.csv' and calculate sampling frequency.
            - Convert quaternion data to rotation matrices and store them.
            - Extract timestamp metadata from 'Command.json'.
        4. Store the processed data and metadata in dictionaries.
        5. Return the combined result.
    """
    # Names of folders where imu data is stored
    imu_subfolders = [item for item in os.listdir(folder) if os.path.isdir(os.path.join(folder, item))]
    dict_keys = [folder.split(' ')[0].lower() for folder in imu_subfolders]

    # If chest key is present, change this to low_back
    dict_keys = ['low_back' if item == 'chest' else item for item in dict_keys]

    # Data dictionary
    imu_data = {}

    # Initialize metadata dict
    imu_metadata = {}
    fs_dict = {}
    date_list = []
    frames_list = []

    # Loop through each subfolder to load quaternion csv files for all imu sensors (back, chair, chest, l_arm, r_arm)
    for subfolder, key in zip(imu_subfolders, dict_keys):
        # Get directory of subfolder
        dir_subfolder = os.path.join(folder, subfolder)

        # Get directory of quaternion file
        file_path = os.path.join(dir_subfolder, 'Quaternion.csv')

        if os.path.exists(file_path):
            # Load csv
            df = pd.read_csv(file_path)

            # Sampling frequency (Hz)
            diff_us = np.diff(df['Timestamp (us)'])                         # Difference between timestamps in microseconds
            diff_s = diff_us / 1000000                                      # Difference in seconds
            avg_diff_s = np.mean(diff_s)                                    # Mean of differences
            fs = 1 / avg_diff_s                                             # Sampling freq in Hz rounded up
            fs_dict[key] = fs

            # Total frames
            frames_list.append(len(df))

            # Create column with rotation matrices
            # Put quaternions in list
            imu_quats = df[['X', 'Y', 'Z', 'W']].values

            # Convert quaternions to rotation matrices in datframe colum rot_matrix
            df['rot_matrix'] = list(R.from_quat(imu_quats).as_matrix())

            # Add df to dictionary
            imu_data[key] = df
        
        else:
            print(f'File not found: {file_path}')

        # Determine metadata of IMU data
        # Get directory for json file for date in metadata
        json_file_path = os.path.join(dir_subfolder, 'Command.json')

        # Load the JSON file
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
        
        date = [entry["time"] for entry in json_data if "time" in entry]
        date_list.append(date)

    # Put metadata in metadata dict
    imu_metadata['imu_names'] = dict_keys
    imu_metadata['date_time'] = date_list
    imu_metadata['total_frames'] = frames_list
    imu_metadata['rotation_type'] = 'Quaternion'
    imu_metadata['coordinate_space'] = 'Global'
    imu_metadata['sampling_freq'] = fs_dict

    return imu_data, imu_metadata

def load_mag_imu_data(folder):
    """
    Loads magnetometer data from each IMU subfolder within the specified folder.

            Parameters:
            folder (str): Path to the main directory containing subdirectories for different IMU sensors.

            Returns:
            tuple:
                - imu_mag_data (dict): A dictionary where each key is a sensor name (e.g., 'low_back', 'chair', 'l_arm'),
                and each value is a pandas DataFrame containing:
                    - 'Timestamp (us)': Timestamps in microseconds.
                    - 'X', 'Y', 'Z': Magnetometer readings in microtesla.
                - fs_dict (dict): A dictionary mapping each sensor name to its computed sampling frequency in Hz.

            Steps:
            1. Identify subdirectories corresponding to individual IMU sensors.
            2. Normalize sensor names (e.g., 'chest' to 'low_back').
            3. For each subfolder:
                - Load 'Magnetometer.csv'.
                - Compute the sampling frequency from timestamp differences.
                - Store the DataFrame in the dictionary under the corresponding key.
            4. Return the data dictionary and the frequency metadata.
    """
    # Names of folders where imu data is stored
    imu_subfolders = [item for item in os.listdir(folder) if os.path.isdir(os.path.join(folder, item))]

    dict_keys = [folder.split(' ')[0].lower() for folder in imu_subfolders]

    # If chest key is present, change this to low_back
    dict_keys = ['low_back' if item == 'chest' else item for item in dict_keys]

    # Data dictionary
    imu_mag_data = {}

    # Initialize metadata dict
    fs_dict = {}

    # Loop through each subfolder to load quaternion csv files for all imu sensors (back, chair, chest, l_arm, r_arm)
    for subfolder, key in zip(imu_subfolders, dict_keys):
        # Get directory of subfolder
        dir_subfolder = os.path.join(folder, subfolder)

        # Get directory of quaternion file
        file_path = os.path.join(dir_subfolder, 'Magnetometer.csv')

        if os.path.exists(file_path):
            # Load csv
            df = pd.read_csv(file_path)

            # Sampling frequency (Hz)
            diff_us = np.diff(df['Timestamp (us)'])                         # Difference between timestamps in microseconds
            diff_s = diff_us / 1000000                                      # Difference in seconds
            avg_diff_s = np.mean(diff_s)                                    # Mean of differences
            fs = 1 / avg_diff_s                                             # Sampling freq in Hz rounded up
            fs_dict[key] = fs

            # Calculate magnetic field magnitude
            df['magnitude'] = np.sqrt(df['X (a.u.)']**2 + df['Y (a.u.)']**2 + df['Z (a.u.)']**2) * 50

            # Add df to dictionary
            imu_mag_data[key] = df

    return imu_mag_data, fs_dict

def load_gyro_imu_data(folder):
    """
    Loads gyroscope data from each IMU subfolder within the specified folder.

            Parameters:
            folder (str): Path to the main directory containing subdirectories for different IMU sensors.

            Returns:
            tuple:
                - imu_gyro_data (dict): A dictionary where each key is a sensor name and the value is a pandas DataFrame
                containing:
                    - 'Timestamp (us)': Timestamps in microseconds.
                    - 'X', 'Y', 'Z': Angular velocity readings in degrees/s or rad/s.
                - fs_dict (dict): A dictionary mapping each sensor name to its computed sampling frequency (Hz).

            Steps:
            1. Identify subfolders corresponding to IMU sensors.
            2. Normalize sensor names (e.g., 'chest' to 'low_back').
            3. For each subfolder:
                - Load 'Gyroscope.csv'.
                - Compute the sampling frequency.
                - Store the data in a dictionary with sensor name as key.
            4. Return the data dictionary and frequency dictionary.
    """
    # Names of folders where imu data is stored
    imu_subfolders = [item for item in os.listdir(folder) if os.path.isdir(os.path.join(folder, item))]
    dict_keys = [folder.split(' ')[0].lower() for folder in imu_subfolders]

    # If chest key is present, change this to low_back
    dict_keys = ['low_back' if item == 'chest' else item for item in dict_keys]

    # Data dictionary
    imu_gyro_data = {}

    # Initialize metadata dict
    fs_dict = {}

    # Loop through each subfolder to load quaternion csv files for all imu sensors (back, chair, chest, l_arm, r_arm)
    for subfolder, key in zip(imu_subfolders, dict_keys):
        # Get directory of subfolder
        dir_subfolder = os.path.join(folder, subfolder)

        # Get directory of quaternion file
        file_path = os.path.join(dir_subfolder, 'Inertial.csv')

        if os.path.exists(file_path):
            # Load csv
            df = pd.read_csv(file_path)

            # Sampling frequency (Hz)
            diff_us = np.diff(df['Timestamp (us)'])                         # Difference between timestamps in microseconds
            diff_s = diff_us / 1000000                                      # Difference in seconds
            avg_diff_s = np.mean(diff_s)                                    # Mean of differences
            fs = 1 / avg_diff_s                                             # Sampling freq in Hz rounded up
            fs_dict[key] = fs

            # Drop accelerometer data
            df = df.drop(columns=['Accelerometer X (g)', 'Accelerometer Y (g)', 'Accelerometer Z (g)'])

            # Add df to dictionary
            imu_gyro_data[key] = df

    return imu_gyro_data, fs_dict

def load_opti_data(file_path):
    """
    Loads optical motion capture data from a CSV file and structures it into a DataFrame of rotation matrices.

            Parameters:
            file_path (str): Path to the CSV file containing the optical data.

            Returns:
            tuple:
                - motion_capture_data (dict): dictionary containing:
                    - 'rigid_body': Dictionary where each key is a rigid body name and the value is a DataFrame with:
                        - 'time': Timestamps in seconds.
                        - 'position (x,y,z)': List of 3D positions for each time step.
                        - 'rotation (x,y,z,w)': List of quaternions representing rotation.
                        - 'mean marker error': Mean marker error for the rigid body.
                    - 'markers': Dictionary where each key is a marker name and the value is a DataFrame with:
                        - 'time': Timestamps in seconds.
                        - 'position (x,y,z)': List of 3D positions for each time step.
                        - 'marker_quality': Quality of the marker data.
                - rotation_matrices (pd.DataFrame): A pandas DataFrame where each column contains a 3x3 rotation matrix for each time step.
                    Each column corresponds to a rigid body and contains the rotation matrices reshaped from quaternion data.
                - rotation_matrices_df (pd.DataFrame): A pandas
                - metadata (dict): Dictionary containing:
                    - 'sampling_freq': Sampling frequency of the motion capture system (Hz).
                    - 'total_frames': Number of frames in the dataset.
                    - 'date_time': Date and time of the recording.
                    - 'rotation_type': Type of rotation representation used (e.g., 'Quaternion').
                    - 'coordinate_space': Reference coordinate space for the data (e.g., 'Global').
                    - 'length_units': Units of measurement for the data.

            Steps:
            1. Load the CSV file using pandas.
            2. Extract rotation matrix data for each rigid body from the file.
            3. Group and reshape rotation matrix components into full 3x3 matrices.
            4. Store each series of matrices as a column in the DataFrame.
            5. Compute sampling frequency and metadata.
            6. Return the rotation matrix data and associated metadata.
    """
    
    with open(file_path, 'r') as file:
        metadata = [next(file) for _ in range(6)]

    metadata_processed = [row.strip().split(',') for row in metadata]

    # Extract metadata containing sampling rate, time and date, etc.
    date_time = metadata_processed[0][11]
    total_frames = int(metadata_processed[0][13])
    rotation_type = metadata_processed[0][17]
    length_units = metadata_processed[0][19]
    coordinate_space = metadata_processed[0][21]

    # # Convert date_time to date time object
    # date_time_raw = date_time.strip().replace('.', ':', 2)
    # date_time_object = datetime.strptime(date_time_raw, '%Y-%m-%d %H:%M:%S.%f')

    # Row 3 specifies data type (rigid body, rigid body marker)
    data_types = metadata_processed[2]

    # Row 4 specifies names of the rb (rolstoel, chair, chest, etc.)
    names = metadata_processed[3]

    # Row 6 specifies the quantity that is measured (rotation, position, mean marker error, etc.)
    units = metadata_processed[5]

    # Read the csv from when the data starts
    data = pd.read_csv(file_path, header=5)

    # Sampling rate (Hz)
    time_string = data['Time (Seconds)']                        # Time data in seconds
    time_diffs = np.diff(time_string)                           # Difference between timesteps
    mean_period = np.mean(time_diffs)                           # Mean difference = period in seconds
    fs = 1 / mean_period                                        # Sampling frequency in Hz

    # Filter columns that are not relevant: rolstoel rigid body, duplicate marker data
    filtered_columns = [
        idx for idx, (dtype, name) in enumerate(zip(data_types, names))
        if dtype != 'Marker' and 'rolstoel' not in name
        ]

    # Subset the data to filtered columns
    filtered_data = data.iloc[:, filtered_columns]

    # Filtered metadata for further processing
    filtered_names = [names[idx] for idx in filtered_columns]
    filtered_data_units = [units[idx] for idx in filtered_columns]

    # # Add filtered names and data types to the filtered_data dataframe for verification
    # filtered_data_with_metadata = filtered_data.copy()
    # filtered_data_with_metadata.columns = pd.MultiIndex.from_tuples(
    #     list(zip(filtered_names, filtered_data_types)),
    #     names=["Name", "Data Type"])

    # Initialize the main dictionary
    motion_capture_data = {
        base_name: {"rigid_body": {}, "markers": {}}
        for base_name in set(name.split(':')[0].strip('"') for name in filtered_names if name and 'rolstoel' and 'Name' not in name)
    }

    for rigid_body in set(filtered_names):
        # Skip rigid body names ta are irrelevant (first column with frames and timestamps)
        if rigid_body in [None, 'Name', '']:
            continue

        # Collect marker data = position data and marker quality of each marker
        if ':' in rigid_body:
            # Separate the name of each marker into the rigid body 
            base_name, marker_name = rigid_body.split(':')
            base_name = base_name.strip('"')
            marker_name = marker_name.strip('"')

            # Determine columns of csv data per marker
            marker_idx = [
                idx for idx, name in enumerate(filtered_names) 
                if name == rigid_body
                ]

            # Add marker position and marker quality data to dictionary
            motion_capture_data[base_name]["markers"][marker_name] = pd.DataFrame({
                "time": data["Time (Seconds)"],
                "position (x,y,z)": filtered_data.iloc[:, marker_idx[0]:marker_idx[3]].values.tolist(),
                "marker_quality": filtered_data.iloc[:, marker_idx[-1]].values.tolist(),
            })

        # Collect rigid body data
        else: 
            rigid_body_idx = [
                idx for idx, name in enumerate(filtered_names)
                if name == rigid_body
            ]

            # Get the columns of the rigi body data and column names (position, rotation and mean marker error)
            rb_data = filtered_data.iloc[:, rigid_body_idx]
            rb_columns = [filtered_data_units[idx] for idx in rigid_body_idx]

            # Extract rigid body-level data
            position_cols = [i for i, col in enumerate(rb_columns) if "Position" in col]
            rotation_cols = [i for i, col in enumerate(rb_columns) if "Rotation" in col]
            mean_error_cols = [i for i, col in enumerate(rb_columns) if "Mean Marker Error" in col]

            # Quat data
            quats = np.array(rb_data.iloc[:, rotation_cols].values.tolist())

            # Print amount of NaN quats
            nan_mask = np.isnan(quats).any(axis=1)
            nan_indices = np.where(nan_mask)[0]
            # print(f"Amount of NaN quaternions (%) in rigid body {rigid_body}: {(len(nan_indices)/len(nan_mask))*100}")
    
            # Convert NaN quats to [0,0,0,1] = no rotation
            quats[nan_indices] = [0, 0, 0, 1]

            # Put position, rotation and mean_marker_error in dataframe
            motion_capture_data[rigid_body]['rigid_body'] = pd.DataFrame({
                'time': data["Time (Seconds)"],
                'position (x,y,z)': rb_data.iloc[:, position_cols].values.tolist(),
                'rotation (x,y,z,w)': rb_data.iloc[:, rotation_cols].values.tolist(), 
                'mean_marker_error': rb_data.iloc[:, mean_error_cols].values.tolist(),
                'rot_matrix': list(R.from_quat(quats).as_matrix()),                                  # Compute rotation matrices
                'NaN_quaternion': nan_mask
            })

    # Additional metadata to return
    metadata = {
        "date_time": date_time,
        "total_frames": total_frames,
        "rotation_type": rotation_type,
        "length_units": length_units,
        "coordinate_space": coordinate_space,
        "sampling_freq": fs
    }

    return motion_capture_data, metadata

def has_extreme_values(matrix):
    """
    Checks whether a matrix contains extreme values based on a threshold.

            Parameters:
            matrix (np.ndarray): Input matrix to be checked (shape: [3,3]).

            Returns:
            bool: True if any value in the matrix is greater than 1000 or less than -1000, False otherwise.

            Steps:
            1. Use numpy to check if any element exceeds the absolute value threshold.
    """
    # Checks if matrix contains values above 1 or below -1
    return np.any(matrix > 1) or np.any(matrix < -1)

def check_R_abnormal(df, matrix_column):
    """
    Identifies and flags rotation matrices that contain NaN values or extreme numerical outliers.

            Parameters:
            df (pd.DataFrame): DataFrame containing a column of rotation matrices.
            matrix_column (str): Name of the column containing 3x3 rotation matrices.

            Returns:
            filtered_indices (np.ndarray): Indices of rows in the DataFrame where the rotation matrices contain NaNs or extreme values.

            Steps:
            1. Iterate through the specified column.
            2. Check each matrix for NaNs and extreme values.
            3. Flag those rows accordingly.
    """
    # Validate if the column exists and contains matrices
    if matrix_column not in df.columns:
        raise ValueError(f"Column '{matrix_column}' does not exist in the DataFrame.")

    # Get indices of rows where the condition is met
    filtered_indices = df.index[df[matrix_column].apply(has_extreme_values)].to_numpy()

    return filtered_indices

def check_R_NaNs(arr):
    """
    Checks whether an array of rotation matrices contains any NaN values.

            Parameters:
            arr (np.ndarray): Array of 3x3 rotation matrices.

            Returns:
            tuple:
                - nan_count (int): Number of NaN elements in the array.
                - nan_percentage (float): Percentage of NaN elements relative to the total number of elements.

            Steps:
            1. Flatten the array and use numpy to check for NaNs.
    """
    total_elements = 0
    nan_count = 0
    
    for sub_array in arr:
        sub_array = np.array(sub_array)  # Ensure it's a NumPy array
        total_elements += sub_array.size
        nan_count += np.isnan(sub_array).sum()
    
    nan_percentage = (nan_count / total_elements) * 100 if total_elements > 0 else 0
    
    return nan_count, nan_percentage

def is_orthogonal(matrix, tol=1e-6):
    """
    Checks whether a given matrix is orthogonal within a specified tolerance.

            Parameters:
            matrix (np.ndarray): A 3x3 matrix to check.
            tol (float): Tolerance for determining orthogonality.

            Returns:
            tuple:
                - identity_check (bool): True if the matrix is orthogonal (i.e., R.T @ R is approximately the identity matrix), False otherwise.
                - determinant_check (bool): True if the determinant of the matrix is approximately 1, False otherwise.

            Steps:
            1. Multiply the matrix by its transpose and check if it equals the identity matrix within the tolerance.
            2. Check if the determinant of the matrix is approximately 1 within the tolerance.
    """
    # Check if R^T * R = I 
    identity_check = np.allclose(np.dot(matrix.T, matrix), np.eye(3), atol=tol)

    # Check determinant
    determinant_check = np.isclose(np.linalg.det(matrix), 1.0, atol=tol)

    return identity_check, determinant_check


# ------------------------------------------------------------------- VALIDATION OF TRUNK MOTION ANALYSIS -------------------------------------------------------------------


def slerp_resample(data, key, current_fs, target_fs):
    """
    Resamples quaternion data for a specific sensor using Spherical Linear Interpolation (SLERP).

            Parameters:
            data (dict): Dictionary with IMU dataframes.
            key (str): Key identifying the specific sensor to resample.
            current_fs (float): Original sampling frequency (Hz).
            target_fs (float): Target sampling frequency (Hz).

            Returns:
            pd.DataFrame: Resampled IMU dataframe with updated quaternion and rotation matrix.

            Steps:
            1. Extract quaternion and timestamp data from the specified sensor dataframe.
            2. Use SLERP to interpolate quaternions to the new time base.
            3. Convert interpolated quaternions to rotation matrices.
            4. Return the updated dataframe.
    """
    # Define time arrays: current time array and target time array
    current_time = np.linspace(0, len(data) / current_fs, num = len(data))
    
    num_samples = round(current_time[-1] * target_fs)           # amount of samples for 120 Hz, rounded to an integer
    resampled_time = np.linspace(0, current_time[-1], num=num_samples)

    actual_fs = num_samples / current_time[-1]          # actual resampled sampling freq, as the num_samples is rounded
    
    # Convert quaternions to rotation objects
    rotation_objects = R.from_quat(data)

    # Perform SLERP
    slerp = Slerp(current_time, rotations=rotation_objects)
    resampled_rotations = slerp(resampled_time)

    # Convert to quaternions
    resampled_quats = resampled_rotations.as_quat()

    # Convert to rotation matrices
    resampled_rot_matrix = resampled_rotations.as_matrix()

    # Add the resampled data to a dataframe
    resamp_data = pd.DataFrame(data={'Timestamp (us)': resampled_time, 'quaternions (x,y,z,w)': list(resampled_quats), 'rot_matrix': list(resampled_rot_matrix)})

    # Perform 2 checks to see if the rotation matrices are still correct
    abnormal_condition = [has_extreme_values(matrix) for matrix in resampled_rot_matrix]
    extreme_check_filtered = [matrix for matrix in resampled_rot_matrix if has_extreme_values(matrix)]

    if extreme_check_filtered:
        print(f'Rotation matrix error: input data[{key}] - amount of resampled matrices in data that contain elements > 1 or < -1 = {len(extreme_check_filtered)}')

    # Check 2: are matrices orthogonal
    orthogonal_condition = [is_orthogonal(matrix) for matrix in resampled_rot_matrix]
    orthogonal_check_filtered = [matrix for matrix in resampled_rot_matrix if not np.all(is_orthogonal(matrix))]

    if orthogonal_check_filtered:
        print(f'Rotation matrix error: input data[{key}] - amount of resampled matrices in data that are not orthogonal = {len(orthogonal_check_filtered)}')

    return resamp_data, actual_fs

def resample_imu_data(imu_data, imu_metadata, opti_metadata):
    """
    Resamples all IMU dataframes to match the optical system's sampling frequency.

            Parameters:
            imu_data (dict): Dictionary of IMU dataframes.
            imu_metadata (dict): Dictionary containing sampling frequency metadata for each IMU sensor.
            opti_metadata (dict): Dictionary containing optical system metadata, including target sampling frequency.

            Returns:
            tuple:
                - resampled_data (dict): Dictionary of IMU dataframes resampled to match the optical system's frequency.
                - resampled_fs (dict): Dictionary mapping each IMU sensor to its resampled sampling frequency.

            Steps:
            1. For each IMU sensor, extract current and target frequencies.
            2. Use SLERP-based resampling for each sensor's quaternion data.
            3. Store resampled data in a new dictionary.
    """
    # Set up resampled dictionary and metadata list
    resampled_data = {}
    resampled_fs = {}

    # Extract opti sampling freq
    opti_fs = opti_metadata['sampling_freq']

    for key in imu_data.keys():
        if key == 'x-imu3':
            continue

        # Extract quaternions
        quats = imu_data[key][['X', 'Y', 'Z', 'W']].values

        # Extract imu sampling freq
        imu_fs = imu_metadata['sampling_freq'][key]

        # Perform SLERP resampling on quats
        resamp_data, resamp_fs = slerp_resample(data=quats, key=key, current_fs=imu_fs, target_fs=opti_fs)

        # Add resampled_fs to resampled_metadata
        resampled_fs[key] = resamp_fs

        # Add resampled data dataframe to resampled data dictionary
        resampled_data[key] = resamp_data

    return resampled_data, resampled_fs

def convert_y_to_z_rotation_matrix(R_y):
    """
    Converts a rotation matrix from Y-axis to Z-axis orientation.

            Parameters:
            R_y (np.ndarray): Rotation matrix in Y-axis frame (3x3).

            Returns:
            R_z (np.ndarray): Equivalent rotation matrix expressed in the Z-axis frame (3x3).

            Steps:
            1. Multiply the input matrix with a transformation matrix that aligns Y to Z orientation.
    """
    if R_y.shape != (3, 3):
        raise ValueError("Input matrix must be a 3x3 rotation matrix.")
    
    # Map elements from R_y to R_z based on the mapping rules
    R_z = np.array([
        [R_y[0, 0], R_y[2, 0], R_y[1, 0]],
        [R_y[0, 2], R_y[2, 2], R_y[1, 2]],
        [R_y[0, 1], R_y[2, 1], R_y[1, 1]]
    ])
    
    return R_z

def preprocess_rot_matrix(R_data, R_0_inv=None, datatype = Literal['imu', 'opti']): 
    """
    Applies preprocessing to rotation matrices based on the selected data type.

            Parameters:
            R_data (np.ndarray): Array of rotation matrices (shape: [N, 3, 3]).
            R_0_inv (np.ndarray): Inverse of the initial rotation matrix.
            datatype (str): Type of preprocessing to apply ('imu' for imu data or 'opti' for mocap data).

            Returns:
            R_final (np.ndarray): Preprocessed rotation matrices (shape: [N, 3, 3]).

            Steps:
            1. If 'relative', multiply each matrix by the inverse of the initial frame.
            2. If 'convert_y_to_z', apply a coordinate transformation.
    """   
    if datatype not in ('imu', 'opti'):
        raise ValueError("datatype must be either 'imu' or 'opti'")

    # Compute R_0 inv
    if R_0_inv is None:
        R_0 = R_data[0]
        R_0_inv = np.linalg.inv(R_0)

    # Determine relative rotation
    R_rel = np.einsum('ijk,kl->ijl', R_data, R_0_inv)

    if datatype == 'opti':
        # Transform opti data into z-axis rotations instead of y-axis rotations
        R_final = np.array([convert_y_to_z_rotation_matrix(rot_matrix) for rot_matrix in R_rel])

    else:  # datatype == 'imu'
        # R_final = np.array([flip_x_rotation(rot_matrix) for rot_matrix in R_rel])
        R_final = R_rel

    return R_final

def rot_matrix_to_euler_angles(R_data, rot_seq = 'XYZ', unwrap=False, degrees=True):
    """
    Converts a sequence of rotation matrices into Euler angles.

            Parameters:
            R_data (np.ndarray): Array of 3x3 rotation matrices (shape: [N, 3, 3]).
            rot_seq (str): Rotation sequence (e.g., 'xyz', 'zyx').
            unwrap (bool): Whether to unwrap the resulting angle sequences to remove discontinuities.
            degrees (bool): If True, returns angles in degrees; otherwise in radians.

            Returns:
            euler (np.ndarray): Array of Euler angles (shape: [N, 3]) for each rotation matrix.

            Steps:
            1. Use scipy to convert rotation matrices to Euler angles using the specified sequence.
            2. Optionally unwrap angles to make them continuous.
            3. Convert angles to degrees if requested.
    """
    if unwrap:
        # Convert to Euler angles in radians since np.unwrap requires radians
        rad_euler = R.from_matrix(R_data).as_euler(rot_seq, degrees=False)

        # Unwrap in radians
        euler_unwrapped = np.unwrap(rad_euler, axis=0)

        if degrees:
            # Convert back to degrees
            euler = np.rad2deg(euler_unwrapped)
        
        euler= euler_unwrapped

    else:
        # Directly convert to Euler angles in degrees
        euler = R.from_matrix(R_data).as_euler(rot_seq, degrees=degrees)

    return euler

def filter_nans_df(df_with_nan_flag: pd.DataFrame, other_df: pd.DataFrame):
    """
    Filters out rows in a DataFrame based on a boolean 'nan_flag' column in another DataFrame.

            Parameters:
            df_with_nan_flag (pd.DataFrame): DataFrame containing a 'nan_flag' column.
            other_df (pd.DataFrame): DataFrame to filter based on 'nan_flag'.

            Returns:
            tuple:
                - df_with_nan_flag_cleaned (pd.DataFrame): Filtered version of `df_with_nan_flag` excluding rows where 'nan_flag' is True.
                - other_df_cleaned (pd.DataFrame): Filtered version of `other_df` excluding rows where 'nan_flag' is True.

            Steps:
            1. Identify indices in `df_with_nan_flag` where 'nan_flag' is False.
            2. Use those indices to select rows from `other_df`.
    """
    # Identify indices to remove
    nan_indices = df_with_nan_flag[df_with_nan_flag['NaN_quaternion'] == True].index

    # Drop rows where 'NaN_quaternion' is True
    df_with_nan_flag_cleaned = df_with_nan_flag.drop(index=nan_indices).reset_index(drop=False)
    other_df_cleaned = other_df.drop(index=nan_indices).reset_index(drop=False)

    return df_with_nan_flag_cleaned, other_df_cleaned

def get_time_shift(signal1, signal2):
    """
    Calculates the optimal time shift between two signals using cross-correlation.

            Parameters:
            signal1 (np.ndarray): First input signal (1D array).
            signal2 (np.ndarray): Second input signal (1D array).

            Returns:
            tuple:
                - time_shift (int): Number of samples by which signal2 should be shifted to best align with signal1.
                - correlation (np.ndarray): Cross-correlation array showing similarity between the two signals at different lags.

            Steps:
            1. Use cross-correlation to determine similarity between the signals.
            2. Identify the lag (index offset) at which the correlation is maximized.
    """
    # Compute cross-correlation
    correlation = correlate(signal1, signal2, mode='full')
    lags = correlation_lags(len(signal1), len(signal2), mode='full')

    # Find the lag with the highest correlation
    max_corr_index = np.argmax(correlation)
    time_shift = lags[max_corr_index]

    return time_shift, correlation

def time_align_signals(signal1, signal2, time_shift):
    """
    Aligns two signals based on a given time shift by trimming or padding with NaNs.

            Parameters:
            signal1 (np.ndarray): Reference signal (1D array).
            signal2 (np.ndarray): Signal to be shifted (1D array).
            time_shift (int): Number of samples to shift signal2.

            Returns:
            tuple:
                - signal1_aligned (np.ndarray): Possibly trimmed version of signal1.
                - signal2_aligned (np.ndarray): Shifted and padded version of signal2.

            Steps:
            1. Shift signal2 based on the time shift direction (positive/negative).
            2. Pad the start or end with NaNs to maintain length consistency.
    """
    # Ensure signals have the same number of dimensions
    if signal1.ndim != signal2.ndim:
        raise ValueError("Both signals must have the same number of dimensions")

    # If time_shift > 0, remove first `time_shift` samples from signal2
    if time_shift > 0:
        aligned_signal1 = signal1[time_shift:len(signal2)+time_shift]  # Trim the beginning of signal2
        aligned_signal2 = signal2[:len(aligned_signal1)]  # Trim to match remaining length

    # If time_shift < 0, remove first `|time_shift|` samples from signal1
    elif time_shift < 0:
        aligned_signal2 = signal2[abs(time_shift):len(signal1)+time_shift]  # Trim the beginning of signal1
        aligned_signal1 = signal1[:len(aligned_signal2)]  # Trim signal2 to match remaining length

    # If no shift, keep both unchanged
    else:
        aligned_signal1 = signal1[:len(signal2)]
        aligned_signal2 = signal2[:len(signal1)]

    return aligned_signal1, aligned_signal2

def time_align_df(df1, df2, time_shift):
    """
    Aligns two DataFrames based on a given sample-wise time shift.

            Parameters:
            df1 (pd.DataFrame): Reference DataFrame.
            df2 (pd.DataFrame): DataFrame to be time-shifted.
            time_shift (int): Number of samples to shift df2.

            Returns:
            tuple:
                - df1_aligned (pd.DataFrame): Possibly trimmed version of df1.
                - df2_aligned (pd.DataFrame): Shifted and padded version of df2.

            Steps:
            1. Use pandas shifting and padding to align df2 relative to df1.
            2. Handle positive and negative shifts differently to preserve indexing.
    """
    # If time_shift > 0, remove first `time_shift` samples from signal2
    if time_shift > 0:
        aligned_df1 = df1.iloc[time_shift:len(df2)+time_shift].reset_index(drop=True)  # Trim the beginning of signal2
        aligned_df2 = df2.iloc[:len(aligned_df1)]  # Trim to match remaining length

    # If time_shift < 0, remove first `|time_shift|` samples from signal1
    elif time_shift < 0:
        aligned_df2 = df2.iloc[abs(time_shift):len(df1)+time_shift].reset_index(drop=True)  # Trim the beginning of signal1
        aligned_df1 = df1.iloc[:len(aligned_df2)]  # Trim signal2 to match remaining length

    # If no shift, keep both unchanged
    else:
        aligned_df1 = df1.iloc[:len(df2)]
        aligned_df2 = df2.iloc[:len(df1)]

    return aligned_df1.copy(), aligned_df2.copy()

def prepocess_and_align_data(resampled_imu_data, opti_data, preproces_method: Literal['ta_first', 'preproc_first'] = 'prepro_first', time_align_method: Literal['part', 'whole', 'tilt_angles', 'fixed'] = 'whole', flipped_fix= False, input_time_shift=None, align_plot=False):
    """
    Preprocesses and aligns IMU and optical data using specified methods.

            Parameters:
            resampled_imu_data (dict): Dictionary of resampled IMU dataframes.
            opti_data (pd.DataFrame): DataFrame of optical rotation matrix data.
            preproces_method (str): Method for preprocessing IMU data ('ta_first', 'preproc_first'): first preprocess rotation matrices or first calculate tilt angles
            time_align_method (str): Method for time alignment ('part', 'whole', 'tilt_angles', 'fixed').
                - part = use only the 360 degree z-rotation part of the movement for cross-correlation to time align
                - whole = use the whole movement for cross-correlation to time align
                - tilt_angles = use tilt angles for cross-correlation to time align
                - fixed = use a fixed time shift (input_time_shift must be provided)
            flipped_fix (bool): Whether to apply a fix for flipped signals.
            input_time_shift (int or None): Optional manual time shift (in samples).
            align_plot (bool): Whether to plot the alignment for visual inspection.

            Returns:
            tuple:
                - af_imu_data (np.ndarray): Preprocessed and aligned Euler angles from IMU data.
                - af_opti_data (np.ndarray): Preprocessed and aligned Euler angles from optical data.

            Steps:
            1. Preprocess IMU and optical rotation matrices based on selected method.
            2. Convert both to Euler angles.
            3. Time-align the angle signals using correlation or manual shift.
            4. Optionally apply signal flipping correction.
            5. Return aligned angles and timestamps.
    """
        
    rot_seq = 'XYZ'
 
    # Get rotation matrix data of chair IMU and RB
    R_opti_chair = np.stack(opti_data['chair']['rigid_body']['rot_matrix'])
    R_imu_chair = np.stack(resampled_imu_data['chair']['rot_matrix'])

    # Preprocessing of R data for imu and opti
    R_opti_chair = preprocess_rot_matrix(R_opti_chair, datatype = 'opti')
    R_imu_chair = preprocess_rot_matrix(R_imu_chair, datatype = 'imu')

    # Convert to Euler angles
    euler_opti = rot_matrix_to_euler_angles(R_opti_chair, rot_seq=rot_seq, unwrap=False, degrees=True)
    euler_imu = rot_matrix_to_euler_angles(R_imu_chair, rot_seq=rot_seq, unwrap=False, degrees=True)

    # Get z-axis angles for time shift
    # imu_z = np.degrees(np.unwrap(euler_imu[:,2], discont=np.radians(90), axis=0))
    # opti_z = np.degrees(np.unwrap(euler_opti[:,2], discont=np.radians(90), axis=0))

    imu_z = euler_imu[:,2]
    opti_z = euler_opti[:,2]

    if time_align_method == 'tilt_angles':
        # Get rotation matrix data of chair IMU and RB
        R_opti_back = preprocess_rot_matrix(np.stack(opti_data['back']['rigid_body']['rot_matrix']), datatype='opti')
        R_imu_back = preprocess_rot_matrix(np.stack(resampled_imu_data['back']['rot_matrix']), datatype = 'imu')

        # Get tilt angles
        tilt_angles_imu = np.array([tilt_angles(R_matrix, degrees=True) for R_matrix in R_imu_back])
        tilt_angles_opti = np.array([tilt_angles(R_matrix, degrees=True) for R_matrix in R_opti_back])

        # # Remove wierd jumps from tilt_angles opti
        # tilt_angles_opti = remove_angle_spikes(tilt_angles_opti, threshold=20, max_plateau_length=30)

        # Determine time shift
        time_shift, correlation = get_time_shift(tilt_angles_imu, tilt_angles_opti)
        print(f'Time shift = {time_shift}')

        # For the plot
        ta_plot_imu = tilt_angles_imu
        ta_plot_opti = tilt_angles_opti

    elif time_align_method == 'part':
        # Use only the 360 degree z-rotation part of the movement to time align
        part_imu_z = imu_z[:int(0.3*len(imu_z))]
        part_opti_z = opti_z[:int(len(part_imu_z))]

        # Determine time shift
        time_shift, correlation = get_time_shift(part_imu_z, part_opti_z)
        print(f'Time shift = {time_shift} samples')

        # For the plot
        ta_plot_imu = imu_z
        ta_plot_opti = opti_z

    elif time_align_method == 'whole':
        # Determine time shift
        time_shift, correlation = get_time_shift(imu_z, opti_z)
        print(f'Time shift = {time_shift} samples')

        # For the plot
        ta_plot_imu = imu_z
        ta_plot_opti = opti_z

    elif time_align_method == 'fixed':
        if input_time_shift is None:
            raise ValueError("extra_value must be provided if time_align_method is fixed")
        
        time_shift = input_time_shift

        # For the plot
        ta_plot_imu = imu_z
        ta_plot_opti = opti_z

    else:
        raise ValueError(f"Unknown time_align_method: {time_align_method}")

    if align_plot:
        # Align the euler IMU and OptiTrack data in time FOR THE PLOT
        aligned_imu, aligned_opti = time_align_signals(ta_plot_imu, ta_plot_opti, time_shift)

        # Plot the original imu and optitrack data together with the aligned imu and optitrack data. Also plot the correlation over the lags
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(ta_plot_imu, label='imu', color='grey')
        ax1.plot(ta_plot_opti, label=f"opti", color='lightgrey')
        ax1.plot(aligned_imu, label=f"aligned imu", color='red', linestyle='--')
        ax1.plot(aligned_opti, label="aligned opti", color='green', linestyle=':')
        # ax1.set_xlabel('Sample')
        ax1.set_ylabel('Angle (deg)')
        ax1.set_title(f'IMU and OptiTrack data, aligned in time with method {time_align_method}')
        ax1.legend()

        ax2.plot(correlation, label='correlation', color='m')
        ax2.set_xlabel('Sample')
        ax2.set_title('Correlation between IMU data and OptiTrack data')
    
    # Loop through the keys of both whole datasets to align the data in time
    imu_keys = list(resampled_imu_data.keys())
    opti_keys = list(opti_data.keys())

    # Check if the keys are the same
    if set(imu_keys) - set(opti_keys):
        only_in_imu_keys = set(imu_keys) - set(opti_keys)
        only_in_opti_keys = set(opti_keys) - set(imu_keys)
        raise ValueError(f"IMU data keys are not the same as Opti data keys: IMU contains {only_in_imu_keys} and Opti contains {only_in_opti_keys}")

    # Set up dictionaries for aligned, preprocessed and filtered (from NaN's) data
    af_imu_data = {}
    af_opti_data = {}

    for key in imu_keys:
        imu_df = resampled_imu_data[key]
        opti_df = opti_data[key]['rigid_body']
        
        if preproces_method == 'ta_first':
            # Time align
            aligned_imu_df, aligned_opti_df = time_align_df(imu_df, opti_df, time_shift=time_shift)

            # Preprocess the rotation matrix data and put in extra df column
            # R data at t=all
            R_imu_t = np.stack(aligned_imu_df['rot_matrix'])
            R_opti_t = np.stack(aligned_opti_df['rot_matrix'])

            # Preprocessing of R data for imu and opti
            prpr_R_imu = preprocess_rot_matrix(R_imu_t, datatype = 'imu')
            prpr_R_opti = preprocess_rot_matrix(R_opti_t, datatype = 'opti')

            # Add preprocessed R_data to colum in df's
            aligned_imu_df.loc[:, 'pr_rot_matrix'] = list(prpr_R_imu)
            aligned_opti_df.loc[:, 'pr_rot_matrix'] = list(prpr_R_opti)

        elif preproces_method == 'preproc_first':
            # Preprocess the rotation matrix data and put in extra df column
            # R data at t=all
            R_imu_t = np.stack(imu_df['rot_matrix'])
            R_opti_t = np.stack(opti_df['rot_matrix'])
        
            # Preprocessing of R data for opti
            prpr_R_imu = preprocess_rot_matrix(R_imu_t, datatype = 'imu')
            prpr_R_opti = preprocess_rot_matrix(R_opti_t, datatype = 'opti')

            # Add preprocessed R_data to colum in df's
            imu_df['pr_rot_matrix'] = list(prpr_R_imu)
            opti_df['pr_rot_matrix'] = list(prpr_R_opti)

            # Align the entire df with data in time
            aligned_imu_df, aligned_opti_df = time_align_df(imu_df, opti_df, time_shift=time_shift)

        else:
            raise ValueError(f"Unknown preproces_method: {preproces_method}")
        
        # Filter the NaN's from the df, keep the old indices
        af_opti_df, af_imu_df = filter_nans_df(aligned_opti_df, aligned_imu_df)
        
        # Add imu and opti data to premade dictionaries
        af_imu_data[key] = af_imu_df
        af_opti_data[key] = af_opti_df

    return af_imu_data, af_opti_data

def tilt_angles(R_matrix, degrees=True):
    """
    Computes forward and lateral tilt angles from a sequence of rotation matrices.

            Parameters:
            R_matrix (np.array): One rotation matrices (shape: [3, 3]).
            degrees (bool): Whether to return angles in degrees or radians.

            Returns:
                - np.ndarray: Tilt angles in radians or degrees, depending on the `degrees` parameter.

            Steps:
            1. Compute arctangent-based tilt angles from matrix axes.
            2. Convert angles to degrees if requested.
    """
    # Z-vector
    z_R_vector = R_matrix[:,2]

    # angle between global z-axis (0,0,1) and rotated z-axis
    angle_rad = np.arccos(np.clip(np.dot(z_R_vector, [0,0,1]), -1.0, 1.0))

    if degrees == True:
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    return angle_rad

def calc_angle_sign(euler_opti_x, euler_opti_y):
    """
    Determines the dominant direction (sign) of an angular movement based on optical Euler angles.

            Parameters:
            euler_opti_x (np.ndarray): Euler angles representing motion in the X direction.
            euler_opti_y (np.ndarray): Euler angles representing motion in the Y direction.

            Returns:
            int: +1 or -1 depending on the dominant direction of motion.

            Steps:
            1. Compute standard deviations of both X and Y angle signals.
            2. Return +1 if X dominates; -1 if Y dominates.
    """
    # Remove beginning offset from data
    euler_opti_x_offset = euler_opti_x - euler_opti_x[0]
    euler_opti_y_offset = euler_opti_y - euler_opti_y[0]

    # Initialize sign lsits
    sign_x = []
    sign_y = []

    # Loop through elements of x-axis and y-axis to determine the sign, append to lists
    for i in range(len(euler_opti_x)):
        if euler_opti_x_offset[i] < 0:
            sign_x.append(-1)

        if euler_opti_x_offset[i] >= 0:
            sign_x.append(1)

        if euler_opti_y_offset[i] < 0:
            sign_y.append(-1)

        if euler_opti_y_offset[i] >= 0:
            sign_y.append(1)
    
    # Calculate the combined signs by adding the elements in the two lists together
    combined_signs = [1 if x == 1 and y == 1 else -1 for x, y in zip(sign_x, sign_y)]

    return sign_x, sign_y, combined_signs

def mad_based_filter(data, thresh=3.5):
    """
    Filters outliers from a signal using the Median Absolute Deviation (MAD) method.

            Parameters:
            data (np.ndarray): Input signal to be filtered.
            thresh (float): Threshold multiplier for determining outliers.

            Returns:
            np.ndarray: Filtered signal with outliers replaced by NaN.

            Steps:
            1. Compute median and MAD.
            2. Identify and replace values that exceed the scaled MAD threshold.
    """
    median = np.median(data, axis=0)
    diff = np.abs(data - median)
    mad = np.median(diff, axis=0)
    modified_z = 0.6745 * diff / mad
    return modified_z < thresh

def compute_angular_velocity(timestamps, angles_deg, time_unit= Literal['us', 's']):
    """
    Calculates angular velocity from a sequence of angle measurements.

            Parameters:
            timestamps (np.ndarray): Timestamps corresponding to angle measurements.
            angles_deg (np.ndarray): Angle values in degrees.
            time_unit (str): Unit of time used for rate computation ('s', 'ms', etc.).

            Returns:
            np.ndarray: Angular velocity values with the same length as input angles.

            Steps:
            1. Compute difference between consecutive angles and timestamps.
            2. Derive angular velocity by dividing angle difference by time difference.
            3. Prepend NaN to maintain array length consistency.
    """
    if time_unit == 'us':
        # Convert microseconds to seconds
        timestamps_s = timestamps / 1e6

    # Timestamps are in seconds
    timestamps_s = timestamps

    # Compute time differences
    dt = np.diff(timestamps_s)  # len = N-1

    # Compute angle differences (absolute to avoid canceling)
    dtheta = np.diff(angles_deg)  # len = N-1

    # Instantaneous angular velocities
    angular_velocities = dtheta / dt  # degrees per second

    # Mean angular velocity
    return angular_velocities

def compute_forward_lean_time(timestamps, angles_deg, threshold_deg=25, time_unit= Literal['us', 's']):
    """
    Computes the duration for which the forward tilt angle exceeds a given threshold.

            Parameters:
            timestamps (np.ndarray): Array of time values corresponding to angle measurements.
            angles_deg (np.ndarray): Forward tilt angles in degrees.
            threshold_deg (float): Threshold angle in degrees to define "leaning forward".
            time_unit (str): Time unit for the output duration ('s', 'ms', etc.).

            Returns:
            float: Total time duration the subject spent leaning forward past the threshold.

            Steps:
            1. Identify indices where angle exceeds the threshold.
            2. Calculate the time difference between those timestamps.
            3. Sum durations to compute total forward-leaning time.
    """
    if time_unit == 'us':
        # Convert microseconds to seconds
        timestamps_s = timestamps / 1e6

    # Timestamps are in seconds
    timestamps_s = timestamps

    # Compute time differences
    dt = np.diff(timestamps_s)  # Length = N - 1

    # Check where angles exceed the threshold
    forward_mask = angles_deg[:-1] > threshold_deg  # Apply to N - 1 for alignment with dt

    # Total time spent in forward lean
    forward_time_s = np.sum(dt[forward_mask])

    # Total time
    total_time_s = timestamps_s[-1] - timestamps_s[0]

    percent_forward = (forward_time_s / total_time_s) * 100

    return forward_time_s, percent_forward

def compute_zero_crossings(angular_velocity):
    """
    Counts the number of zero crossings in an angular velocity signal.

            Parameters:
            angular_velocity (np.ndarray): Angular velocity signal (1D array).

            Returns:
            int: Total number of zero crossings in the signal.

            Steps:
            1. Compute the sign of the angular velocity.
            2. Count how many times the sign changes between consecutive samples.
    """
    zero_crossings = np.where(np.diff(np.sign(angular_velocity)))[0]
    return len(zero_crossings)

def bland_altman_subplot(ax, data1, data2, title='', scatter_color='blue'):
    """
    Generates a Bland-Altman plot on the given Matplotlib axis.

            Parameters:
            ax (matplotlib.axes.Axes): The axis to plot on.
            data1 (np.ndarray): First set of measurement data.
            data2 (np.ndarray): Second set of measurement data to compare.
            title (str): Title for the plot.
            scatter_color (str): Color for scatter points.

            Returns:
            dict: Dictionary containing Bland-Altman statistics (bias, LoA, std deviation).

            Steps:
            1. Compute the mean and difference between the two datasets.
            2. Calculate bias, limits of agreement (LoA), and standard deviation.
            3. Plot the results on the provided axis.
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff)

    ax.scatter(mean, diff, alpha=0.5, color=scatter_color)
    ax.axhline(md, color='gray', linestyle='--', label=f'Mean = {md:.2f}')
    ax.axhline(md + 1.96*sd, color='red', linestyle='--')
    ax.axhline(md - 1.96*sd, color='red', linestyle='--')

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Mean (°)')
    ax.set_ylabel('Difference (°)')
    ax.grid(True)

    return {
        'bias': md,
        'loa_upper': md + 1.96*sd,
        'loa_lower': md + 1.96*sd,
        'std_diff': sd
    }

def bland_altman_plot(imu_data, opti_data, plot=False, title="Bland-Altman Plot"):
    """
    Creates a full Bland-Altman plot comparing IMU and optical data for each axis.

            Parameters:
            imu_data (np.ndarray): IMU-based angle data (shape: [N, 3]).
            opti_data (np.ndarray): Optical-based angle data (shape: [N, 3]).
            plot (bool): Whether to display the plot.
            title (str): Title of the figure.

            Returns:
            dict: Dictionary containing Bland-Altman statistics for each axis (bias, LoA, std deviation).

            Steps:
            1. Compute residuals between IMU and optical data.
            2. Calculate mean and difference for each axis.
            3. Create subplots for each axis and compute Bland-Altman statistics.
            4. Plot the comparison if requested.
    """
    imu_data = np.asarray(imu_data)
    opti_data = np.asarray(opti_data)
    
    # Mean and difference
    mean = (imu_data + opti_data) / 2
    diff = imu_data - opti_data
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff)

    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    # Plotting
    if plot == True:
        plt.figure(figsize=(8, 5))
        plt.scatter(mean, diff, alpha=0.6)
        plt.axhline(mean_diff, color='red', linestyle='--', label=f'Mean Bias = {mean_diff:.2f}')
        plt.axhline(loa_upper, color='gray', linestyle='--', label=f'+1.96 SD = {loa_upper:.2f}')
        plt.axhline(loa_lower, color='gray', linestyle='--', label=f'-1.96 SD = {loa_lower:.2f}')
        plt.title(title)
        plt.xlabel('Mean of IMU and MoCap angles')
        plt.ylabel('Difference (IMU - MoCap)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Return stats if you want them for reporting
    return {
        'bias': mean_diff,
        'loa_upper': loa_upper,
        'loa_lower': loa_lower,
        'std_diff': sd_diff
    }

def concordance_correlation_coefficient(x, y):
    """
    Calculates the Concordance Correlation Coefficient (CCC) between two signals.

            Parameters:
            x (np.ndarray): First array of measurements.
            y (np.ndarray): Second array of measurements.

            Returns:
            float: Concordance correlation coefficient between the two signals.

            Steps:
            1. Use the pingouin library to compute CCC from paired data arrays.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))

    ccc = (2 * cov_xy) / (var_x**2 + var_y**2 + (mean_x - mean_y)**2)
    return ccc

def remove_residual_outliers(angles_imu, angles_opti, method='iqr', threshold=1.5):
    """
    Removes outlier samples based on residuals between IMU and optical signals using a specified method.

            Parameters:
            angles_imu (np.ndarray): IMU-based angle measurements (shape: [N, 3]).
            angles_opti (np.ndarray): Optical-based angle measurements (shape: [N, 3]).
            method (str): Outlier detection method ('iqr').
            threshold (float): Threshold for identifying outliers.

            Returns:
            tuple:
                - angles_imu_filtered (np.ndarray): Filtered IMU angles with outliers removed or replaced.
                - angles_opti_filtered (np.ndarray): Filtered optical angles with outliers removed or replaced.

            Steps:
            1. Calculate residuals (differences) between IMU and optical signals.
            2. Identify outliers using the MAD method.
            3. Replace outlier entries with NaNs or remove them.
    """
    # Step 1: Compute residuals
    residuals = angles_imu - angles_opti

    # Step 2: Detect outliers using IQR
    if method == 'iqr':
        q1 = np.percentile(residuals, 25)
        q3 = np.percentile(residuals, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        mask = (residuals >= lower_bound) & (residuals <= upper_bound)

    # Optional: Z-score method
    elif method == 'zscore':
        z_scores = (residuals - np.mean(residuals)) / np.std(residuals)
        mask = np.abs(z_scores) < threshold

    else:
        raise ValueError("Invalid method. Use 'iqr' or 'zscore'")

    # Step 3: Apply mask to both angle series
    residuals_filtered = residuals[mask]

    return mask, residuals_filtered

def preprocess_single_trial(subject, test, speed, test_date, dir):
    """
    Loads and processes IMU and optical data for a single subject trial.

            Parameters:
            subject (str): Subject identifier.
            test (str): Name of the test performed.
            speed (str): Speed condition ('hs' or 'ls').
            test_date (str): Date of the test session.
            dir (str): Root directory containing all data.

            Returns:
            tuple:
                - final_angles_imu (np.ndarray): Preprocessed IMU angle data.
                - final_angles_opti (np.ndarray): Preprocessed optical angle data.

            Steps:
            1. Load IMU and optical data from specified directories.
            2. Resample and align IMU data to match optical sampling frequency.
            3. Preprocess and align both signals using default methods.
            4. Return aligned rotation matrices for IMU and 3D mocap data.
    """
    imu_dir = f'test_{test_date}/{subject}/imu/{subject}_{test}_{speed}'
    opti_dir = f'test_{test_date}/{subject}/opti'
    imu_folder = os.path.join(dir, imu_dir)
    opti_file = os.path.join(dir, opti_dir, f'{subject}_{test}_{speed}.csv')

    if not os.path.isdir(imu_folder):
        print(f"[SKIP] IMU folder not found for {subject}, {test}, {speed}: {imu_folder}")
        return None, None
    if not os.path.isfile(opti_file):
        print(f"[SKIP] MOCAP file not found for {subject}, {test}, {speed}: {opti_file}")
        return None, None

    imu_data, imu_meta = load_imu_data(imu_folder)
    opti_data, opti_meta = load_opti_data(opti_file)

    resampled_imu_data, _ = resample_imu_data(imu_data, imu_meta, opti_meta)

    if (subject, test, speed) in part_combinations:
        aligned_imu, aligned_opti = prepocess_and_align_data(resampled_imu_data, opti_data, 'ta_first', 'part', False)
    elif (subject, test, speed) in ta_combinations:
        aligned_imu, aligned_opti = prepocess_and_align_data(resampled_imu_data, opti_data, 'ta_first', 'tilt_angles', False)
    elif (subject, test, speed) in fixed_combinations:
        shift = fixed_time_shift[subject]
        aligned_imu, aligned_opti = prepocess_and_align_data(resampled_imu_data, opti_data, 'ta_first', 'fixed', input_time_shift=shift, align_plot=False)
    else:
        aligned_imu, aligned_opti = prepocess_and_align_data(resampled_imu_data, opti_data, 'ta_first', 'whole', False)

    return aligned_imu, aligned_opti

def calculate_tilt_angles(aligned_imu, aligned_opti, subject, test, speed, rigid_body):
    """
    Calculates forward and lateral tilt angles for a given rigid body segment.

            Parameters:
            aligned_imu (np.ndarray): Aligned IMU rotation matrix or angle data.
            aligned_opti (np.ndarray): Aligned optical rotation matrix or angle data.
            subject (str): Subject identifier.
            test (str): Name of the test.
            speed (str): Speed condition.
            rigid_body (str): Name of the rigid body (e.g., 'low_back').

            Returns:
            tuple:
                - angles_imu (np.ndarray): IMU-based tilt angles.
                - angles_opti (np.ndarray): Optical-based tilt angles.

            Steps:
            1. Select appropriate axis or matrix component.
            2. Compute tilt using trigonometric relationships.
            3. Return results for both data sources.
    """
    R_imu = np.stack(aligned_imu[rigid_body]['pr_rot_matrix'])
    R_opti = np.stack(aligned_opti[rigid_body]['pr_rot_matrix'])

    tilt_imu = np.array([tilt_angles(Ri, degrees=True) for Ri in R_imu])
    tilt_opti = np.array([tilt_angles(Ri, degrees=True) for Ri in R_opti])

    if (subject, test, speed, rigid_body) in flipped_combinations:
        idx = np.nonzero(tilt_opti)[0][0]
        tilt_opti = tilt_imu[idx] + abs(tilt_opti - tilt_opti[idx])

    if (subject, test, speed, rigid_body) in offset_combinations:
        tilt_opti = abs(tilt_opti - np.mean(tilt_opti))

    return tilt_imu, tilt_opti

def compute_trunk_metrics(subject, test, speed, rigid_body, final_angles_imu, final_angles_opti, imu_time, opti_time):
    """
    Computes trunk motion metrics from aligned IMU and optical angle data.

            Parameters:
            subject (str): Subject identifier.
            test (str): Test name.
            speed (str): Speed condition.
            rigid_body (str): Rigid body name.
            final_angles_imu (np.ndarray): IMU-derived angles.
            final_angles_opti (np.ndarray): Optical-derived angles.
            imu_time (np.ndarray): IMU timestamps.
            opti_time (np.ndarray): Optical timestamps.

            Returns:
            dict: Dictionary of computed trunk metrics. Trunk metrics include: mean_angle, range of motion, standard deviation, mean angular velocity, zero crossing counts, percent leaning forward > 25 deg.

            Steps:
            1. Extract forward and lateral tilt.
            2. Compute max ROM, mean tilt, forward lean duration, zero crossing counts.
            3. Package metrics into a structured dictionary.
    """
    imu_rom = np.ptp(final_angles_imu)
    opti_rom = np.ptp(final_angles_opti)

    return {
        'subject': subject,
        'test': test,
        'speed': speed,
        'sensor': rigid_body,
        'mean_angle_imu': np.mean(final_angles_imu),
        'mean_angle_opti': np.mean(final_angles_opti),
        'rom_imu': imu_rom,
        'rom_opti': opti_rom,
        'std_imu': np.std(final_angles_imu),
        'std_opti': np.std(final_angles_opti),
        'mean_ang_vel_imu': np.mean(compute_angular_velocity(imu_time, final_angles_imu, 'us')),
        'mean_ang_vel_opti': np.mean(compute_angular_velocity(opti_time, final_angles_opti, 's')),
        'zero_crossings_imu': compute_zero_crossings(compute_angular_velocity(imu_time, final_angles_imu, 'us')),
        'zero_crossings_opti': compute_zero_crossings(compute_angular_velocity(opti_time, final_angles_opti, 's')),
        'percent_forward_imu': compute_forward_lean_time(imu_time, final_angles_imu, 15, 'us')[1],
        'percent_forward_opti': compute_forward_lean_time(opti_time, final_angles_opti, 15, 's')[1]
    }
    
def compute_validation_metrics(final_angles_imu, final_angles_opti):
    """
    Computes validation metrics comparing IMU-derived angles to optical angles.

            Parameters:
            final_angles_imu (np.ndarray): IMU angle data (shape: [N, 3]).
            final_angles_opti (np.ndarray): Optical angle data (shape: [N, 3]).

            Returns:
            dict: Dictionary containing validation metrics. Validation metrics include: 
                - mean absolute error (mae)
                - root mean square error (rmse)
                - concordance correlation coefficient (CCC)
                - intraclass correlation coefficient (ICC [3,1])
                - Bland-Altman statistics

            Steps:
            1. Determine the residuals between IMU and optical angles.
            2. Compute mean absolute error (MAE) and root mean square error (RMSE).
            3. Calculate concordance correlation coefficient (CCC).
            4. Compute intraclass correlation coefficient (ICC [3,1]).
            5. Generate a Bland-Altman plot and compute its statistics.
            6. Compute Bland-Altman statistics: bias, upper/lower limits of agreement, std deviation.
    """
    residuals = final_angles_imu - final_angles_opti
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    ccc = concordance_correlation_coefficient(final_angles_imu, final_angles_opti)

    n = len(final_angles_imu)
    df = pd.DataFrame({
        'target': np.tile(np.arange(n), 2),
        'rater': ['IMU'] * n + ['MOCAP'] * n,
        'score': np.concatenate([final_angles_imu, final_angles_opti])
    })
    icc_val = pg.intraclass_corr(data=df, targets='target', raters='rater', ratings='score')
    icc_3_1 = icc_val[icc_val['Type'] == 'ICC3'].iloc[0]['ICC']
    ba = bland_altman_plot(final_angles_imu, final_angles_opti)

    return {
        'mae': mae,
        'rmse': rmse,
        'CCC': ccc,
        'ICC [3,1]': icc_3_1,
        'Bland-Altman': ba
    }
    
def aggregate_metrics(subjects, subjects_pro, tests, speeds, rigid_bodies, data_dir, test_date_pro, test_date_amateur):
    """
    Aggregates performance and validation metrics across multiple subjects, tests, and conditions.

            Parameters:
            - subjects (list): List of subject identifiers.
            - subjects_pro (list): List of professional subject identifiers.
            - tests (list): List of test types or labels.
            - speeds (list): List of test speeds or conditions.
            - rigid_bodies (list): List of rigid body labels (e.g., body parts with IMU).
            - data_dir (str): Path to the directory containing motion data files.
            - test_date_pro (str): Test date for professional subjects.
            - test_date_amateur (str): Test date for amateur subjects.

            Returns:
            - pd.DataFrame: Summary metrics for each valid trial.
            - pd.DataFrame: Validation metrics including statistical agreement between IMU and MOCAP.
            - dict: Dictionary mapping each rigid body to a list of all IMU tilt angles across trials.
            - dict: Dictionary mapping each rigid body to a list of all MOCAP tilt angles across trials.

            Main Steps:
            1. Initialize containers for metrics and angle data.
            2. Loop through all valid combinations of subject, test, speed, and rigid body.
            3. Preprocess data for each trial using `preprocess_single_trial`.
            4. Compute trunk tilt angles from IMU and MOCAP using `calculate_tilt_angles`.
            5. Remove outliers using IQR-based filtering (`remove_residual_outliers`).
            6. Compute summary trunk movement metrics (`compute_trunk_metrics`).
            7. Compute validation statistics (e.g., RMSE, correlation) using `compute_validation_metrics`.
            8. Store all results and angle data for post-hoc analysis.
    """
    summary_metrics = []
    validation_metrics = []

    imu_all_angles = {rb: [] for rb in rigid_bodies}
    opti_all_angles = {rb: [] for rb in rigid_bodies}

    for subject in subjects:
        test_date = test_date_pro if subject in subjects_pro else test_date_amateur

        for test in tests:
            for speed in speeds:
                for rb in rigid_bodies:
                    if (subject, test, speed) in non_valid_combinations:
                        continue
                    if (subject, speed, test, rb) in bad_LB_combinations:
                        continue

                    imu_data, opti_data = preprocess_single_trial(subject, test, speed, test_date, data_dir)
                    tilt_imu, tilt_opti = calculate_tilt_angles(imu_data, opti_data, subject, test, speed, rb)

                    mask, _ = remove_residual_outliers(tilt_imu, tilt_opti, method='iqr')
                    tilt_imu = tilt_imu[mask]
                    tilt_opti = tilt_opti[mask]

                    imu_time = np.stack(imu_data[rb]['Timestamp (us)'])[mask]
                    opti_time = np.stack(opti_data[rb]['time'])[mask]

                    summary = compute_trunk_metrics(subject, test, speed, rb, tilt_imu, tilt_opti, imu_time, opti_time)
                    summary_metrics.append(summary)

                    validation = compute_validation_metrics(tilt_imu, tilt_opti)
                    validation.update({'subject': subject, 'test': test, 'speed': speed, 'sensor': rb})
                    validation_metrics.append(validation)
                    
                    imu_all_angles[rb].extend(tilt_imu.tolist())
                    opti_all_angles[rb].extend(tilt_opti.tolist())

    return pd.DataFrame(summary_metrics), pd.DataFrame(validation_metrics), imu_all_angles, opti_all_angles

def normality_check_plot(imu_angles, opti_angles, rigid_body):
    """
    Performs statistical normality checks and visual comparisons of IMU vs MOCAP trunk tilt angles for a given rigid body.

            Parameters:
            - imu_angles (dict): Dictionary of IMU tilt angles, keyed by rigid body label.
            - opti_angles (dict): Dictionary of MOCAP tilt angles, keyed by rigid body label.
            - rigid_body (str): Label of the rigid body to analyze (e.g., "lower_back").

            Returns:
            - None: Displays diagnostic plots and prints normality statistics to console.

            Main Steps:
            1. Extract and flatten tilt angle arrays for IMU and MOCAP.
            2. Compute residuals (difference between IMU and MOCAP).
            3. Filter residuals using IQR method to remove outliers.
            4. Perform and print results of D’Agostino normality test (p-value, skewness, kurtosis).
            5. Generate and display plots:
                - Histograms for IMU and MOCAP tilt angles.
                - Scatter plot showing correlation.
                - Boxplot and histogram of residuals.
                - Bland-Altman plot for agreement analysis.
    """
    palette = ["#ea5545", "#f46a9b", "#eb790e", "#eae850", "#87bc45", "#395D0E", "#27aeef", "#1a33c0", "#b33dc6"]
    
    imu_data = np.array(imu_angles[rigid_body]).flatten()
    opti_data = np.array(opti_angles[rigid_body]).flatten()

    residuals = imu_data - opti_data
    mask, filtered_residuals = remove_residual_outliers(imu_data, opti_data, method='iqr', threshold=1.5)

    print("Normality Results for", rigid_body)
    for label, data in zip(['IMU', 'OPTI', 'RESIDUALS'], [imu_data, opti_data, residuals]):
        stat, p = normaltest(data)
        print(f"{label}: p={p:.4f}, skew={skew(data):.2f}, kurtosis={kurtosis(data):.2f}")

    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f'Data Comparison for {rigid_body.capitalize()}', fontsize=16)
    # fig.suptitle(f'Distribution analysis for upper back (T3-T4)', fontsize=16)

    # IMU histogram
    axs[0, 0].hist(imu_data, bins=100, alpha=0.9, color=palette[5])
    axs[0, 0].set_title('IMU trunk tilt angles histogram')
    axs[0, 0].set_xlabel('Trunk tilt angle (°)')
    axs[0, 0].set_ylabel('Frequency')

    # OptiTrack histogram
    axs[0, 1].hist(opti_data, bins=100, alpha=0.9, color=palette[5])
    axs[0, 1].set_title('MOCAP trunk tilt angles histogram')
    axs[0, 1].set_xlabel('Trunk tilt angle (°)')
    axs[0, 1].set_ylabel('Frequency')

    # Correlation plot
    axs[0, 2].scatter(imu_data, opti_data, alpha=0.9, s=2, color=palette[1])
    axs[0, 2].set_title('IMU vs MOCAP')
    axs[0, 2].set_xlabel('IMU trunk tilt angle (°)')
    axs[0, 2].set_ylabel('MOCAP trunk tilt angle (°)')
    axs[0, 2].grid(True)
    corr_coef = np.corrcoef(imu_data, opti_data)[0, 1]
    axs[0, 2].annotate(f'r = {corr_coef:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=12, ha='left', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    # Residuals boxplot
    axs[1, 0].boxplot(residuals)
    axs[1, 0].set_title('Boxplot of IMU and MOCAP differences')
    axs[1, 0].set_ylabel('Difference (°)')
    axs[1, 0].grid(True)

    # Residuals histogram
    axs[1, 1].hist(residuals, bins=100, alpha=0.9, color=palette[0])
    axs[1, 1].set_title('Histogram of IMU and MOCAP differences')
    axs[1, 1].set_xlabel('Trunk tilt angle (°)')
    axs[1, 1].set_ylabel('Frequency')

    # Bland-Altman plot
    BA_results = bland_altman_subplot(axs[1, 2], imu_data, opti_data, title='Bland-Altman plot for IMU and 3D MOCAP', scatter_color=palette[6])
    print(f'Bland-Altman results: bias = {BA_results['bias']}, LoA = {BA_results['loa_upper']}-{BA_results['loa_lower']}, std = {BA_results['std_diff']}')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


# ------------------------------------------------------------------- TRUNK MOTION, PERFORMANCE, AND CLASSIFICATION ANALYSIS -------------------------------------------------------------------


def wheel_imu_misalignment_correction(ximu3_rot_matrix, correction = False, unwrap=True):
    # Convert rotation matrix to Euler angles
    ximu3_euler = R.from_matrix(ximu3_rot_matrix).as_euler('xyz', degrees=True)

    if correction == True:
        # Determine median x-angles where y-angles are 0, 90, -90 and 180
        median_0 = np.nanmedian(ximu3_euler[np.where(np.abs(ximu3_euler[:, 2])<1),1])
        median_180 = np.nanmedian(ximu3_euler[np.where(np.abs(ximu3_euler[:, 2])>179),1])
        median_90 = np.nanmedian(ximu3_euler[np.where((ximu3_euler[:, 2]>89)&(ximu3_euler[:, 2]<91)),1])
        median_min_90 = np.nanmedian(ximu3_euler[np.where((ximu3_euler[:, 2]<-89)&(ximu3_euler[:, 2]>-91)),1])

        # Misalignment 
        mis_alignemnt_y = (median_0-median_180)/2
        mis_alignemnt_x = (median_min_90-median_90)/2

        # Apply misalignment correction
        r_y = R.from_euler('y', mis_alignemnt_y, degrees=True).as_matrix()
        cor_y_matrix = np.matmul(np.linalg.inv(r_y), ximu3_rot_matrix)

        r_x = R.from_euler('x', mis_alignemnt_x, degrees=True).as_matrix()
        cor_matrix = np.matmul(np.linalg.inv(r_x), cor_y_matrix)

        euler = R.from_matrix(cor_matrix).as_euler('xyz', degrees=True)

    else:
        euler = ximu3_euler

    if unwrap == True:
        unwrapped_euler = np.unwrap(euler, discont=np.radians(360), axis=0)
        
        return unwrapped_euler
    
    return euler    

def remove_angle_spikes(signal, threshold=30, max_plateau_length=10):
    signal = np.array(signal, dtype=float)
    clean_signal = signal.copy()
    
    outlier_indices = []
    i = 1
    while i < len(signal):
        diff = np.abs(signal[i] - signal[i - 1])
        if diff > threshold:
            # Spike detected
            start = i
            end = i

            # Look ahead for a plateau of similar values
            while end + 1 < len(signal) and end - start < max_plateau_length:
                if np.abs(signal[end + 1] - signal[end]) < 1e-3:
                    end += 1
                else:
                    break

            outlier_indices.extend(range(start, end + 1))
            i = end + 1  # Skip ahead past plateau
        else:
            i += 1

    # Interpolate over outliers
    all_indices = np.arange(len(signal))
    good_indices = np.setdiff1d(all_indices, outlier_indices)

    if len(good_indices) > 1:
        interp_func = interp1d(good_indices, clean_signal[good_indices], kind='linear', fill_value="extrapolate")
        clean_signal[outlier_indices] = interp_func(outlier_indices)

    return clean_signal

def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

def compute_wheel_ang_vel(wheel_roll, fs):
    # Calculate wheel rotational speed in deg/s over time
    wheel_ang_speed = np.diff(wheel_roll) * fs

    # Filter high jumps out
    wheel_ang_speed_clean = remove_angle_spikes(wheel_ang_speed, threshold=10)
    wheel_ang_speed_ma = moving_average(wheel_ang_speed_clean, window_size=15)

    return wheel_ang_speed_ma

def compute_chair_ang_vel_acc(wheel_pivot, frame_pivot, wheel_ang_vel, fs):
    # Calculate the wheelchair pivot speed in deg/s over time
    chair_ang_speed = np.diff(wheel_pivot) * fs

    # Determine gyro offset and remove this from wheelchair pivot rotation
    stable_ind = np.where(abs(wheel_ang_vel)<5) and np.where(abs(chair_ang_speed)<5)
    chair_ang_speed_offset = np.mean(chair_ang_speed[stable_ind])
    chair_ang_speed_cor = chair_ang_speed - chair_ang_speed_offset

    # Filter high jumps out from the velocity
    chair_ang_speed_cor_clean = remove_angle_spikes(chair_ang_speed_cor, threshold=10)
    chair_ang_speed_cor_ma = moving_average(chair_ang_speed_cor_clean, window_size=15)

    # Compute the angular acceleration of the chair
    chair_ang_acc = np.diff(chair_ang_speed_cor) * fs

    # Filter out jumps from the acceleration
    chair_ang_acc_clean = remove_angle_spikes(chair_ang_acc, threshold=10)

    return chair_ang_speed_cor_ma, chair_ang_acc_clean

def compute_wheel_tan_vel(wheel_ang_speed, wheel_diameter):
    wheel_tan_speed = np.deg2rad(wheel_ang_speed) * (wheel_diameter/2)

    # Filter out high jumps
    wheel_tan_speed_clean = remove_angle_spikes(wheel_tan_speed, threshold=0.2)
    wheel_tan_speed_ma = moving_average(wheel_tan_speed_clean, window_size=15)

    return wheel_tan_speed_ma

def compute_chair_lin_vel_acc(wheel_tan_speed, chair_ang_speed, wheel_base, fs):
    # Make sure lengths are the same
    min_len = min(len(wheel_tan_speed), len(chair_ang_speed))

    wheel_tan_speed = wheel_tan_speed[:min_len]
    chair_ang_speed = chair_ang_speed[:min_len]
    
    # Compute the chair linear velocity
    chair_linear_speed =  np.sum([wheel_tan_speed,(np.tan(np.deg2rad(chair_ang_speed)/fs)*wheel_base/2)*fs],axis=0)

    # Filter out jumps from velocity
    chair_linear_speed_clean = remove_angle_spikes(chair_linear_speed, threshold=0.3)
    chair_linear_speed_ma = moving_average(chair_linear_speed_clean, window_size=15)

    # Compute the chair linear acceleration
    chair_linear_acc = np.diff(chair_linear_speed) * fs

    # Filter out jumps from acceleration
    chair_linear_acc_clean = remove_angle_spikes(chair_linear_acc, threshold=30)

    return chair_linear_speed_ma, chair_linear_acc_clean

def compute_performance(wheel_euler, frame_euler, imu_fs, wheel_diameter=0.6, wheel_base=0.75):
    # Wheel Euler angles over time
    wheel_roll = wheel_euler[:,0]
    wheel_pitch = wheel_euler[:,1]
    wheel_pivot = wheel_euler[:,2]
    
    # Frame Euler angles over time
    frame_roll = frame_euler[:,0]
    frame_pitch = frame_euler[:,1]
    frame_pivot = frame_euler[:,2]

    # Calculate wheel rotational speed in deg/s over time
    wheel_ang_vel = compute_wheel_ang_vel(wheel_roll, imu_fs)
    max_wheel_ang_vel = np.max(np.abs(wheel_ang_vel))                                   # Max angular velocity of the wheel (deg/s)
    avg_wheel_ang_vel = np.mean(np.abs(wheel_ang_vel))                                  # Mean angular velocity of the wheel (deg/s)

    # Calculate the wheelchair pivot speed in deg/s over time
    chair_ang_vel, chair_ang_acc = compute_chair_ang_vel_acc(wheel_pivot, frame_pivot, wheel_ang_vel, imu_fs)
    max_chair_ang_vel = np.max(np.abs(chair_ang_vel))                                   # Max angular velocity of the wheelchair (deg/s)
    avg_chair_ang_vel = np.mean(np.abs(chair_ang_vel))                                  # Mean angular velocity of the wheelchair (deg/s)
    
    max_chair_ang_acc = np.max(np.abs(chair_ang_acc))                                   # Max angular acceleration of the wheelchair (deg/s^2)
    avg_chair_ang_acc = np.mean(np.abs(chair_ang_acc))                                  # Mean angular acceleration of the wheelchair (deg/s^2)

    # Calculate the tangential speed of the wheel in m/s over time (is this supposed to be the tangential vel?)
    wheel_tan_vel = compute_wheel_tan_vel(wheel_ang_vel, wheel_diameter)
    max_wheel_tan_vel = np.max(np.abs(wheel_tan_vel))                                   # Max tangential velocity of the wheel (m/s)
    avg_wheel_tan_vel = np.mean(np.abs(wheel_tan_vel))                                  # Mean tangential velocity of the wheel (m/s)

    # Calculate the linear speed of the wheelchair base in m/s
    chair_linear_vel, chair_linear_acc = compute_chair_lin_vel_acc(wheel_tan_vel, chair_ang_vel, wheel_base, imu_fs)
    max_chair_linear_vel = np.max(np.abs(chair_linear_vel))                             # Max linear velocity of the wheelchair (m/s)
    avg_chair_linear_vel = np.mean(np.abs(chair_linear_vel))                            # Mean linear velocity of the wheelchair (m/s)
    
    max_chair_linear_acc = np.max(np.abs(chair_linear_acc))                             # Max linear acceleration of the wheelchair (m/s^2)
    avg_chair_linear_acc = np.mean(np.abs(chair_linear_acc))                            # Mean linear acceleration of the wheelchair (m/s^2)

    # Max velocities and avg velocities
    max_perf_metrics = (max_wheel_ang_vel, max_chair_ang_vel, max_chair_ang_acc, max_wheel_tan_vel, max_chair_linear_vel, max_chair_linear_acc)
    avg_perf_metrics = (avg_wheel_ang_vel, avg_chair_ang_vel, avg_chair_ang_acc, avg_wheel_tan_vel, avg_chair_linear_vel, avg_chair_linear_acc)
    
    # Return performance signals
    perf_signals = {
        'wheel_ang_vel': wheel_ang_vel,
        'chair_ang_vel': chair_ang_vel,
        'chair_ang_acc': chair_ang_acc,
        'wheel_tan_vel': wheel_tan_vel,
        'chair_linear_vel': chair_linear_vel,
        'chair_linear_acc': chair_linear_acc
    }

    return max_perf_metrics, avg_perf_metrics, perf_signals

def aggregate_performance_metrics(subjects_pro, tests, speeds, rigid_bodies, data_dir, test_date_pro):
    """
    Aggregates trunk and wheelchair performance metrics across all valid trials.

    Returns:
        - pd.DataFrame: Summary of trunk and wheelchair motion metrics per trial.
    """
    summary_metrics_list = []

    for subject in subjects_pro:
        test_date = test_date_pro

        for test in tests:
            for speed in speeds:
                if (subject, test, speed) in non_valid_combinations:
                    continue

                # === Load and preprocess data ===
                imu_data, opti_data = preprocess_single_trial(subject, test, speed, test_date, data_dir)
                if imu_data is None or opti_data is None:
                    continue

                # === Wheelchair performance metrics ===
                imu_raw_data, imu_meta = load_imu_data(
                    os.path.join(data_dir, f'test_{test_date}/{subject}/imu/{subject}_{test}_{speed}')
                )

                ximu3_rot_matrix = np.stack(imu_raw_data['x-imu3']['rot_matrix'])
                corrected_euler = wheel_imu_misalignment_correction(ximu3_rot_matrix)
                ximu3_fs = imu_meta['sampling_freq']['x-imu3']

                chair_rot_matrix = np.stack(imu_raw_data['chair']['rot_matrix'])
                chair_euler = R.from_matrix(chair_rot_matrix).as_euler('xyz', degrees=True)

                max_perf, avg_perf, _ = compute_performance(corrected_euler, chair_euler, imu_fs=ximu3_fs)

                for rb in rigid_bodies:
                    if (subject, speed, test, rb) in bad_LB_combinations:
                        continue

                    tilt_imu, tilt_opti = calculate_tilt_angles(imu_data, opti_data, subject, test, speed, rb)
                    mask, _ = remove_residual_outliers(tilt_imu, tilt_opti, method='iqr')
                    final_angles_imu = tilt_imu[mask]

                    # # Euler angles for residual sign correction
                    # euler_imu = R.from_matrix(np.stack(imu_data[rb]['pr_rot_matrix'])).as_euler('XYZ', degrees=True)[mask]
                    # sign_x, _, _ = calc_angle_sign(euler_imu[:, 0], euler_imu[:, 1])
                    # sign_final_angles_imu = final_angles_imu * sign_x

                    # IMU time
                    imu_time = np.stack(imu_data[rb]['Timestamp (us)'])[mask]
                    imu_ang_vel = compute_angular_velocity(imu_time, final_angles_imu, 'us')

                    forward_time_s_imu, percent_forward_imu = compute_forward_lean_time(imu_time, final_angles_imu, threshold_deg=15, time_unit='us')

                    # === Save trial metrics ===
                    summary_metrics_list.append({
                        'subject': subject,
                        'test': test,
                        'speed': speed,
                        'sensor': rb,

                        # Trunk motion
                        'min_angle_imu': np.min(final_angles_imu),
                        'max_angle_imu': np.max(final_angles_imu),
                        'rom_imu': np.ptp(final_angles_imu),
                        'mean_angle_imu': np.mean(final_angles_imu),
                        'mean_ang_vel_imu': np.mean(imu_ang_vel),
                        'max_ang_vel_imu': np.max(imu_ang_vel),
                        'std_imu': np.std(final_angles_imu),
                        'forward_time_imu_s': forward_time_s_imu,
                        'percent_forward_imu': percent_forward_imu,
                        'zero_crossings_imu': compute_zero_crossings(imu_ang_vel),

                        # Wheelchair performance (max)
                        'max_chair_ang_vel': max_perf[1],
                        'max_chair_ang_acc': max_perf[2],
                        'max_chair_linear_vel': max_perf[4],
                        'max_chair_linear_acc': max_perf[5],

                        # Wheelchair performance (avg)
                        'avg_chair_ang_vel': avg_perf[1],
                        'avg_chair_ang_acc': avg_perf[2],
                        'avg_chair_linear_vel': avg_perf[4],
                        'avg_chair_linear_acc': avg_perf[5],
                    })

    return pd.DataFrame(summary_metrics_list)
