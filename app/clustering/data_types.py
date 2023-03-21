"""Data type definitions."""
from enum import Enum


class ModelEnum(Enum):
    """Enumerate models which can be used to generate embeddings."""

    MINILM = "all-MiniLM-L6-v2"
    ADA2 = "text-embedding-ada-002"


class ClusteringModelEnum(Enum):
    """Enumerate clustering models which can be used to form clusters."""

    SPECTRAL = "SpectralClustering"
    KMEANS = "KMeans"
