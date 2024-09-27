import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance

class BalancedKMeans:
    def __init__(self, k, max_iter=300):
        self.k = k
        self.max_iter = max_iter
        self.kmeans = KMeans(n_clusters=k, max_iter=max_iter)
        self.target_size = None
        self.clusters = None

    def fit(self, X):
        n = len(X)
        self.target_size = n // self.k
        labels = self.kmeans.fit_predict(X)
        self.clusters = {i: [] for i in range(self.k)}
        
        for i, label in enumerate(labels):
            self.clusters[label].append(i)
        
        # Balance the clusters
        for cluster in self.clusters.values():
            while len(cluster) > self.target_size:
                # Move items from larger clusters to smaller clusters
                excess = cluster.pop()
                # Find closest cluster with room
                closest_cluster = min(self.clusters.keys(), key=lambda c: distance.euclidean(X[excess], self.kmeans.cluster_centers_[c]) if len(self.clusters[c]) < self.target_size else np.inf)
                self.clusters[closest_cluster].append(excess)
        
        balanced_labels = np.zeros_like(labels)
        for cluster_label, items in self.clusters.items():
            for item in items:
                balanced_labels[item] = cluster_label
        
        return balanced_labels

    def predict(self, X):
        # Predict clusters for new data points
        predictions = self.kmeans.predict(X)
        return predictions

def balanced_kmeans(X, k, max_iter=300):
    model = BalancedKMeans(k, max_iter)
    labels = model.fit(X)
    return labels, model

# Example usage
#X = np.random.rand(100, 2)  # 100 data points in 2D space
#k = 5  # Number of clusters
#labels, model = balanced_kmeans(X, k)
#print("Training labels:", labels)

# Predict on new data
#X_new = np.random.rand(20, 2)  # 20 new data points
#new_labels = model.predict(X_new)
#print("Predicted labels for new data:", new_labels)