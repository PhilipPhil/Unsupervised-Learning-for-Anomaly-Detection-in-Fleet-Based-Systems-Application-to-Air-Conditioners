from tools.Cluster import Cluster
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from collections import Counter

class HierarchicalClustering(Cluster):
    affinity='euclidean'
    linkage='ward'
    distance_threshold=15
    
    def cluster(self, x_row: pd.DataFrame) -> list:   
        clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=self.distance_threshold).fit(x_row)
        labels = clustering.labels_
        color_counts = Counter(labels)
        anomaly_score = [1 - color_counts[color] / len(labels) for color in labels]
        return anomaly_score