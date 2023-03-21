"""Generate embeddings with sentence transformers or through openai api"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import umap
from kneed import KneeLocator
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from tqdm import tqdm

from data_types import ClusteringModelEnum, ModelEnum
from embedding_generator import EmbeddingGenerator, read_csv_data


class Clusterer(EmbeddingGenerator):
    """Generate and visualize clusters."""

    def __init__(self, model_name: ModelEnum) -> None:

        super().__init__(model_name=model_name)
        self.cluster_labels = None
        self.reduced_embedding = None
        self.computed_metrics: Optional[List[Dict[str, Any]]] = None
        self.normalized_metrics: Optional[List[Dict[str, Any]]] = None

    def cluster_data(
        self, clustering_model: ClusteringModelEnum, data: torch.Tensor, n_classes: int
    ) -> None:
        """Generate cluster labels for embeddings
        Args:
            clustering_model (ClusteringModelEnum): the clustering model to use
        """
        if clustering_model == ClusteringModelEnum.SPECTRAL:
            clustering = SpectralClustering(n_clusters=n_classes)
            cluster_labels = clustering.fit_predict(X=data)
            self.cluster_labels = cluster_labels
        elif clustering_model == ClusteringModelEnum.KMEANS:
            clustering = KMeans(n_clusters=n_classes, n_init="auto")
            cluster_labels = clustering.fit_predict(X=data)
            self.cluster_labels = cluster_labels

    def reduce_dimension(self, data: torch.Tensor) -> None:
        """Reduce embedding dimension with UMAP for visualization.
        Args:
            data (torch.Tensor): data to visualize.
        """
        reducer = umap.UMAP()
        pca = PCA(n_components=36)
        data = pca.fit_transform(data)
        umap_embedding = reducer.fit_transform(data)
        self.reduced_embedding = umap_embedding

    def compute_metrics(self, data: torch.Tensor):
        """Calculate different metrics for varying number of clusters.
        Args:
            data (torch.Tensor): data to cluster
        """
        computed_metrics = []
        max_clusters = int(len(data) / 3)
        for k in tqdm(range(2, max_clusters)):
            spectral = SpectralClustering(n_clusters=k)
            spectral.fit(data)
            vrc = metrics.calinski_harabasz_score(data, spectral.labels_)
            silhouette = metrics.silhouette_score(
                data, spectral.labels_, metric="euclidean"
            )
            dbi = metrics.davies_bouldin_score(data, spectral.labels_)
            inertia = calculate_inertia(data, spectral.labels_)
            computed_metrics.append(
                {
                    "k": k,
                    "vrc": vrc,
                    "silhouette": silhouette,
                    "inertia": inertia,
                    "dbi": dbi,
                }
            )
        self.computed_metrics = computed_metrics

    def normalize_metrics(self) -> None:
        """Normalize computed metrics to range [0,1]
        Raises:
            ValueError: Will raise ValueError if metrics have not been computed
        """
        if not self.computed_metrics:
            raise ValueError("Metrics not computed!")
        self.normalized_metrics = copy.deepcopy(self.computed_metrics)
        for key in ["vrc", "silhouette", "inertia", "dbi"]:
            values = [item[key] for item in self.computed_metrics]
            max_val = max(values)
            min_val = min(values)
            normalized_values = [
                (value - min_val) / (max_val - min_val) for value in values
            ]
            for i, value in enumerate(normalized_values):
                self.normalized_metrics[i][key] = value

    def elbow_method(self) -> None:
        """Calculate elbow point using kneed.
        Raises:
            ValueError: raises ValuError if metrics not computed
        """
        if not self.computed_metrics:
            raise ValueError("Metrics not computed!")
        num_of_clusters = [value["k"] for value in self.computed_metrics]
        inertia = [value["inertia"] for value in self.computed_metrics]
        kneedle = KneeLocator(
            num_of_clusters,
            inertia,
            curve="convex",
            direction="decreasing",
            S=1.0,
        )
        kneedle.plot_knee_normalized()
        if kneedle.elbow != None:
            plt.title(f"Elbow point at {round(kneedle.elbow, 2)}")
        plt.savefig("data/elbow_plot.png")

    def plot_metrics(self) -> None:
        """Plot computed metrics. A separate plot is generated where
        the metrics are normalized to the range [0,1].
        Raises:
            ValueError: Raises ValueError if metrics not computed.
        """
        if not self.computed_metrics:
            raise ValueError("Metrics not computed!")
        computed_metrics_df = pd.DataFrame(self.computed_metrics)
        computed_metrics_df.plot(
            x="k",
            y=["vrc", "silhouette", "inertia", "dbi"],
            figsize=(15, 5),
        )
        plt.title("Metrics plot")
        plt.savefig("data/metrics_plot.png")
        self.normalize_metrics()
        normalized_metrics_df = pd.DataFrame(self.normalized_metrics)
        normalized_metrics_df.plot(
            x="k",
            y=["vrc", "silhouette", "inertia", "dbi"],
            figsize=(15, 5),
        )
        plt.title("Normalized metrics plot")
        plt.savefig("data/normalized_metrics_plot.png")


def calculate_inertia(data: torch.Tensor, labels: list) -> float:
    """Calculate inertia for arbitrary labeling of data.
    Args:
        data (torch.Tensor): data to cluster
        labels (list): labels from clustering
    Returns:
        float: inertia of clustering
    """
    unique_labels = list(set(labels))
    data_df = pd.DataFrame(data.numpy())
    inertia = 0.0
    for label in unique_labels:
        cluster_data = data_df.loc[[x == label for x in labels]]
        cluster_centroid = cluster_data.sum(axis=0)
        cluster_centroid = cluster_centroid / np.sqrt(
            cluster_centroid @ cluster_centroid
        )
        for i in cluster_data.index.values:
            cluster_point = cluster_data.loc[i]
            inertia += squared_euclidian_distance(cluster_centroid, cluster_point)
    return inertia


def squared_euclidian_distance(vec1: pd.DataFrame, vec2: pd.DataFrame) -> float:
    """Return squared euclidian distance of two points.
    Args:
        vec1 (pd.DataFrame): first vector
        vec2 (pd.DataFrame): second vector
    Returns:
        float: squared euclidean distance
    """
    diff_vec = vec1 - vec2
    return diff_vec @ diff_vec


if __name__ == "__main__":

    df = read_csv_data()

    clusterer = Clusterer(model_name=ModelEnum.ADA2)
    clusterer.load_embeddings(df)
    clusterer.cluster_data(
        clustering_model=ClusteringModelEnum.SPECTRAL,
        data=clusterer.embeddings,
        n_classes=len(df["jobs"].unique()),
    )
    clusterer.reduce_dimension(clusterer.embeddings)
    clusterer.compute_metrics(clusterer.embeddings)
    print(clusterer.computed_metrics)
    clusterer.plot_metrics()
    clusterer.elbow_method()
