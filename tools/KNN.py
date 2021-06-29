from tools.Cluster import Cluster
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class KNN(Cluster):
    def cluster(self, x_row: pd.DataFrame) -> list:
        nbrs = NearestNeighbors(n_neighbors=len(x_row)//2, algorithm='brute', metric='manhattan').fit(x_row)
        distances, indices = nbrs.kneighbors(x_row)
        labbels = distances.mean(axis=1)
        anomaly_score = (labbels - min(labbels)) / (max(labbels) - min(labbels))
        return anomaly_score.tolist()