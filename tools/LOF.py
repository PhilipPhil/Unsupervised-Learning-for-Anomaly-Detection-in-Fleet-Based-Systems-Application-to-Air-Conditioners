from tools.Cluster import Cluster
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

class LOF(Cluster):
    metric ='euclidean'

    def cluster(self, x_row: pd.DataFrame) -> list:
        x_row = (x_row - x_row.min()) / (x_row.max() - x_row.min())
        clf = LocalOutlierFactor(n_neighbors=len(x_row)//2, algorithm='brute', metric=self.metric).fit(x_row)
        labels = clf.negative_outlier_factor_ * (-1)
        anomaly_score = labels
        # anomaly_score = labels / 20 # Normalize
        return anomaly_score.tolist()