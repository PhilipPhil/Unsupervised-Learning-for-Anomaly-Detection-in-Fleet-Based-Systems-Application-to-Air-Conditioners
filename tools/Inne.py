from tools.Cluster import Cluster
import pandas as pd
from anomatools.models import iNNE

class Inne(Cluster):    
    def cluster(self, x_row: pd.DataFrame) -> list:
        detector = iNNE().fit(x_row)
        anomaly_score = detector.predict_proba(x_row, method='linear')[:,1]
        return anomaly_score.tolist()