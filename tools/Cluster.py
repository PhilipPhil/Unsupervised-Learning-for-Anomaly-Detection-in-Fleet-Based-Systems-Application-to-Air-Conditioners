import pandas as pd
import os
from sklearn.metrics import *
from typing import Tuple

from abc import ABC, abstractmethod

class Cluster(ABC):
    
    def __init__(self):
        self.ACs = []
        for ac_name in os.listdir('./data/ACs'):
            ac_i = pd.read_pickle('./data/ACs/'+ac_name)
            self.ACs.append(ac_i)
        self.n_rows = len(self.ACs[0])
        super().__init__()

    def get_X_Y(self, row: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = []
        for AC in self.ACs:
            AC = pd.DataFrame(AC)
            data.append(AC.iloc[row,:].tolist())
        data = pd.DataFrame(data).reset_index(drop=True)
        X = data.iloc[:,1:]
        Y = data.iloc[:,0]
        
        return X, Y.astype(int).tolist()
        
    def cluster_all(self) -> Tuple[list, list]:
        
        anomaly_score, y = [], []
        
        for row in range(self.n_rows):
            if row % (self.n_rows//20) == 0: print(str(int(row/self.n_rows * 100)) + '% complete')
                
            x_row, y_row = self.get_X_Y(row)
            
            assert sum(y_row) < len(y_row)//2

            anomaly_score_row = self.cluster(x_row)
            anomaly_score += anomaly_score_row
            y += y_row
        print('100% complete')   
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
        return accuracy, precision, recall, f1, yhat
    
    @abstractmethod
    def cluster(self, x_row: pd.DataFrame) -> list:
        pass