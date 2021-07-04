from tools.Cluster import Cluster
import pandas as pd
from sklearn.svm import OneClassSVM

class SVMOneClass(Cluster):
    def cluster(self, x_row: pd.DataFrame) -> list:
        clf = OneClassSVM(kernel='rbf', gamma='auto').fit(x_row)
        labels = clf.decision_function(x_row)
        anomaly_score = 1 - (labels - min(labels))/(max(labels) - min(labels))
        return anomaly_score.tolist()