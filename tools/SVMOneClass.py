from tools.Cluster import Cluster
import pandas as pd
from sklearn.svm import OneClassSVM

class SVMOneClass(Cluster):

    nu = 0.5

    def cluster(self, x_row: pd.DataFrame) -> list:
        clf = OneClassSVM(kernel='rbf', gamma='auto', nu=self.nu).fit(x_row)
        labels = clf.score_samples(x_row) * (-1)
        anomaly_score = labels
        return anomaly_score.tolist()