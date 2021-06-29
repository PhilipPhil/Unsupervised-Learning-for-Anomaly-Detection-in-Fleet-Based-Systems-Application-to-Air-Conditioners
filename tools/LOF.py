from tools.Cluster import Cluster
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

class LOF(Cluster):
    def cluster(self, x_row: pd.DataFrame) -> list:
        clf = LocalOutlierFactor(n_neighbors=len(x_row)//2, algorithm='brute', metric='manhattan').fit(x_row)
        labels = clf.negative_outlier_factor_
        anomaly_score = 1 - (labels - min(labels))/(max(labels) - min(labels))
        return anomaly_score.tolist()