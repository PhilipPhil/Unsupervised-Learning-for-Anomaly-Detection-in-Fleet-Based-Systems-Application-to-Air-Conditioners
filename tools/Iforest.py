from tools.Cluster import Cluster
import pandas as pd
from sklearn.ensemble import IsolationForest

class Iforest(Cluster):
    def cluster(self, x_row: pd.DataFrame) -> list:
        clf = IsolationForest().fit(x_row)
        labels = clf.decision_function(x_row) * (-1)
        anomaly_score = labels
        return anomaly_score.tolist()