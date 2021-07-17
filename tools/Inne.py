from tools.Cluster import Cluster
import pandas as pd
from anomatools.models import iNNE

class Inne(Cluster):
    n_members = 200
    sample_size = 32
    metric='euclidean'
    def cluster(self, x_row: pd.DataFrame) -> list:
        detector = iNNE(n_members=self.n_members, sample_size=self.sample_size, metric=self.metric).fit(x_row)
        anomaly_score = detector.scores_
        return anomaly_score.tolist()