from contextlib import suppress
import pandas as pd
import numpy as np
import os
from sklearn.metrics import *
from typing import Tuple

from abc import ABC, abstractmethod

class Cluster(ABC):
    """
    feature is the variables being used in the model
    options are
        'all' : all variables
        'real' : real power
        'reactive' :  reactive power
        'harmonic' : total harmonic distortion
    """

    def __init__(self, feature = 'all', suppress_progress = False):
        self.ACs = []
        for ac_name in os.listdir('./data/ACs'):
            ac_i = pd.read_pickle('./data/ACs/'+ac_name)
            self.ACs.append(ac_i)
        self.n_rows = len(self.ACs[0])
        self.feature = feature
        self.suppress_progress = suppress_progress
        super().__init__()

    def get_X_Y(self, row: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = []
        for AC in self.ACs:
            AC = pd.DataFrame(AC)
            data.append(AC.iloc[row,:].tolist())
        data = pd.DataFrame(data).reset_index(drop=True)

        X = data.iloc[:,1:]
        if self.feature == 'real':
            X = X.iloc[:,:1500]
        elif self.feature == 'reactive':
            X = X.iloc[:,1500:3000]
        elif self.feature == 'harmonic':
            X = X.iloc[:,3000:]
            
        Y = data.iloc[:,0]
        
        return X, Y.astype(int).tolist()
        
    def cluster_all(self) -> Tuple[list, list]:
        
        anomaly_score, y = [], []
        
        for row in range(self.n_rows):
            if row % (self.n_rows//4) == 0 and not self.suppress_progress: print(str(int(row/self.n_rows * 100)) + '% complete')
                
            x_row, y_row = self.get_X_Y(row)
            
            assert sum(y_row) < len(y_row)//2

            anomaly_score_row = self.cluster(x_row)
            anomaly_score += anomaly_score_row
            y += y_row
        if not self.suppress_progress: print('100% complete')

        self.anomaly_score = np.array(anomaly_score)
        self.y = np.array(y)
        return anomaly_score, y
    
    def auc_score(self, anomaly_score: list, y: list) -> int:
        fpr, tpr, thresholds = roc_curve(y, anomaly_score)
        AUC = auc(fpr, tpr)
        return AUC
    
    def score(self, anomaly_score: list, y: list, threashold: float) -> Tuple[float, float, float, float, list]:
        yhat = [ 1 if y_i >= threashold else 0 for y_i in anomaly_score]
        accuracy = accuracy_score(y, yhat)
        precision = precision_score(y, yhat)
        recall = recall_score(y, yhat)
        f1 = f1_score(y, yhat)
        return accuracy, precision, recall, f1

    def get_scores(self):
        AUC = self.auc_score(self.anomaly_score, self.y)
        print('AUC score: ' + str(AUC))
        anomaly_score = (self.anomaly_score - self.anomaly_score.min()) / (self.anomaly_score.max() - self.anomaly_score.min())
        print('threashold, accuracy, precision, recall, f1')
        for threashold in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]:
            print(threashold, self.score(anomaly_score, self.y, threashold))

    @abstractmethod
    def cluster(self, x_row: pd.DataFrame) -> list:
        pass