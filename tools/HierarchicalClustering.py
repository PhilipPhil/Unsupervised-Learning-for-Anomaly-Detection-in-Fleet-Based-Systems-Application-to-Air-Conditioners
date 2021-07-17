from tools.Cluster import Cluster
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.cluster import AgglomerativeClustering

class HierarchicalClustering(Cluster):
    affinity='euclidean'
    linkage='average'
    distance_threshold = 5 # Known from expert knowlege
    
    def cluster(self, x_row: pd.DataFrame) -> list:
        clustering = AgglomerativeClustering(n_clusters=None, affinity=self.affinity, linkage=self.linkage, distance_threshold=self.distance_threshold).fit(x_row)
        labels = clustering.labels_
        color_counts = Counter(labels)
        anomaly_score = [1 - color_counts[color] / len(labels) for color in labels]
        return anomaly_score