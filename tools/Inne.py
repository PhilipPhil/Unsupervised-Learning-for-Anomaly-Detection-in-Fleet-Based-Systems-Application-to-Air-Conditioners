from tools.Cluster import Cluster
import pandas as pd
from anomatools.models import iNNE

class Inne(Cluster):
    sample_size = 10
    metric='euclidean'
    def cluster(self, x_row: pd.DataFrame) -> list:
        detector = iNNE(sample_size=self.sample_size, metric=self.metric).fit(x_row)
        anomaly_score = detector.scores_
        return anomaly_score.tolist()