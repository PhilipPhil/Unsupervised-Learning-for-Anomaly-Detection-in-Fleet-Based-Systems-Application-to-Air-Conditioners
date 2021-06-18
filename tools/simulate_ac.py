import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import seed
from random import random
from random import randint

def generate_ac_data(dir: str, state: str, AC_COUNT = 10, ALPHA = 0.9, N = 400) -> None:
    data = pd.read_pickle(dir)

    # Remove first 3 minutes, and last 30 seconds
    healthy = data[data[0]==0][6:][:-1]
    faulty = data[data[0]==1][6:][:-1]

    # Fill missing values with row mean
    healthy = healthy.T.fillna(healthy.iloc[:,1:].mean(axis=1)).T
    faulty = faulty.T.fillna(faulty.iloc[:,1:].mean(axis=1)).T

    for i in range(AC_COUNT):
        ac_data = []
        for _ in range(N):
            alpha = random()
            if alpha > ALPHA:
                index = randint(0,len(faulty)-1)
                ac_data.append(faulty.iloc[index,:])
            else:
                index = randint(0,len(healthy)-1)
                ac_data.append(healthy.iloc[index,:])
        ac_data = pd.DataFrame(ac_data).reset_index(drop=True)
        ac_data.to_pickle("./ACs/ac_" + state + "_" + str(i) + ".pkl")

if __name__ == "__main__":
    generate_ac_data("data/harmonic_low_18_cold_fullyOpen.pkl","harmonic_low_18_cold_fullyOpen")