from tools.Cluster import Cluster
import pandas as pd
from anomatools.models import iNNE

class Inne(Cluster):
    n_members = 200
    sample_size = 32
    def cluster(self, x_row: pd.DataFrame) -> list:
        detector = iNNE(n_members=self.n_members, sample_size=self.sample_size).fit(x_row)
        anomaly_score = detector.predict_proba(x_row, method='linear')[:,1]
        return anomaly_score.tolist()