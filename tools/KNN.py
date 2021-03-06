from tools.Cluster import Cluster
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class KNN(Cluster):
    metric ='euclidean'
    k = 0.5

    def cluster(self, x_row: pd.DataFrame) -> list:
        x_row = (x_row - x_row.min()) / (x_row.max() - x_row.min())
        nbrs = NearestNeighbors(n_neighbors=int(len(x_row) * self.k), algorithm='brute', metric=self.metric).fit(x_row)
        distances, indices = nbrs.kneighbors(x_row)
        labbels = distances.mean(axis=1)
        anomaly_score = labbels
        return anomaly_score.tolist()