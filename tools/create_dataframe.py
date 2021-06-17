import pandas as pd
import numpy as np
import os

import ml_tools.io.utils.files
import ml_tools.io.utils.conversion
import ml_tools.batch_processing.scripts.process_synced_files as psf

SAMPLE_LENGTH = 300000

def create_current_dataframe(no_fault_folder: str, fault_folder: str, name: str) -> None:
    data = []
    for dir, pre_dir, state in [(os.listdir(no_fault_folder), no_fault_folder, 0), (os.listdir(fault_folder), fault_folder, 1)]:
        for filename in dir:
            sample_data = np.fromfile(pre_dir + '/' + filename, dtype='uint16').reshape(-1,2)
            current = sample_data[:,1]
            if len(current) == SAMPLE_LENGTH:
                data.append([state] + current.tolist())

    df_new = pd.DataFrame(data)
    df_new.to_pickle("./data/current_" + name + ".pkl")

def create_power_dataframe(no_fault_folder: str, fault_folder: str, name: str) -> None:
    data = []
    for dir, pre_dir, state in [(os.listdir(no_fault_folder), no_fault_folder, 0), (os.listdir(fault_folder), fault_folder, 1)]:
        for filename in dir:
            sample_data = np.fromfile(pre_dir + '/' + filename, dtype='uint16').reshape(-1,2)
            
            power = ml_tools.io.utils.conversion.uint16_to_cycle_avg_power(sample_data)
            data.append([state] + power[:,0].tolist())
            # data.append(power[:,0].tolist())

    df_new = pd.DataFrame(data)
    df_new.to_pickle("./data/power_" + name + ".pkl")

def create_reactive_power_dataframe(no_fault_folder: str, fault_folder: str, name: str) -> None:
    data = []
    for dir, pre_dir, state in [(os.listdir(no_fault_folder), no_fault_folder, 0), (os.listdir(fault_folder), fault_folder, 1)]:
        for filename in dir:
            sample_data = np.fromfile(pre_dir + '/' + filename, dtype='uint16').reshape(-1,2)
            
            power = ml_tools.io.utils.conversion.uint16_to_cycle_avg_power(sample_data)
            data.append([state] + power[:,1].tolist())
            # data.append(power[:,1].tolist())

    df_new = pd.DataFrame(data)
    df_new.to_pickle("./data/reactive_" + name + ".pkl")

def create_harmonic_distortion_dataframe(no_fault_folder: str, fault_folder: str, name: str) -> None:
    data = []
    for dir, pre_dir, state in [(os.listdir(no_fault_folder), no_fault_folder, 0), (os.listdir(fault_folder), fault_folder, 1)]:
        for filename in dir:
            sample_data = np.fromfile(pre_dir + '/' + filename, dtype='uint16').reshape(-1,2)
            
            power = ml_tools.io.utils.conversion.uint16_to_cycle_avg_power(sample_data)
            data.append([state] + power[:,2].tolist())
            # data.append(power[:,2].tolist())

    df_new = pd.DataFrame(data)
    df_new.to_pickle("./data/harmonic_" + name + ".pkl")

if __name__ == "__main__":
    no_fault_folder = './raw_data/low_18_cold_fullyOpen_noFault'
    fault_folder = './raw_data/low_18_cold_fullyOpen_airFilterFault'
    name = 'low_18_cold_fullyOpen'
    create_current_dataframe(no_fault_folder, fault_folder, name)
    create_power_dataframe(no_fault_folder, fault_folder, name)
    create_reactive_power_dataframe(no_fault_folder, fault_folder, name)
    create_harmonic_distortion_dataframe(no_fault_folder, fault_folder, name)