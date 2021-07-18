from tools.Cluster import Cluster
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

class LOF(Cluster):
    metric ='euclidean'
    k = 0.5
    
    def cluster(self, x_row: pd.DataFrame) -> list:
        x_row = (x_row - x_row.min()) / (x_row.max() - x_row.min())
        clf = LocalOutlierFactor(n_neighbors=int(len(x_row) * self.k), algorithm='brute', novelty=True, metric=self.metric).fit(x_row)
        labels = clf.decision_function(x_row) *(-1)
        anomaly_score = labels
        return anomaly_score.tolist()