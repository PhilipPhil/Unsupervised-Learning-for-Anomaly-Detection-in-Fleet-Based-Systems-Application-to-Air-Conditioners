import pandas as pd
import numpy as np
import os
from random import random
from random import randint

from tools.Calculations import Calculations

class Simulate:

    master_dir = "./data/master_dataframe.pkl"
    no_fault_folder = './raw_data/healthy'
    fault_folder = './raw_data/faulty'

    def __init__(self,  ALPHA = 0.9, N_acs = 10, N_rows = 100) -> None:
        self.ALPHA = ALPHA
        self.N_acs = N_acs
        self.N_rows = N_rows

    def create_master_df(self) -> None:
        data = []
        for dir, pre_dir, state in [(os.listdir(self.no_fault_folder), self.no_fault_folder, 0), (os.listdir(self.fault_folder), self.fault_folder, 1)]:
            for filename in dir:
                data_row = np.fromfile(pre_dir + '/' + filename, dtype='uint16').reshape(-1,2)
                calc = Calculations(data_row)
                data.append([state] + calc.real_power.tolist() + calc.reactive_power.tolist() + calc.thd.tolist())

        df_new = pd.DataFrame(data)
        df_new.to_pickle(self.master_dir)

    def generate_ac_data(self) -> None:
        data = pd.read_pickle(self.master_dir)
        normalized_df = (data - data.min()) / (data.max() - data.min())

        healthy = normalized_df[normalized_df[0]==0]
        faulty = normalized_df[normalized_df[1]==1]

        for i in range(self.N_acs):
            ac_data = []
            fault_counts = 1
            for _ in range(self.N_rows):
                alpha = random()
                if alpha > self.ALPHA and fault_counts < self.N_acs//2:
                    fault_counts += 1
                    index = randint(0,len(faulty)-1)
                    ac_data.append(faulty.iloc[index,:])
                else:
                    index = randint(0,len(healthy)-1)
                    ac_data.append(healthy.iloc[index,:])
            
            ac_data = pd.DataFrame(ac_data).reset_index(drop=True)
            ac_data.to_pickle("./ACs/ac_" + str(i) + ".pkl")
