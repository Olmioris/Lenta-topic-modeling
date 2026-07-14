"""
Clustering configuration for BERTopic.

Responsibilities:
- Create UMAP and HDBSCAN instances with given parameters.
"""

import logging
from umap import UMAP
from hdbscan import HDBSCAN

logger = logging.getLogger(__name__)


def build_umap(n_neighbors: int = 15,
               n_components: int = 5,
               min_dist: float = 0.0,
               metric: str = "cosine",
               random_state: int = 42) -> UMAP:
    """
    Build UMAP model for dimensionality reduction.
    """
    logger.info("Building UMAP model...")
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    return umap_model


def build_hdbscan(min_cluster_size: int = 30,
                  min_samples: int = 10,
                  metric: str = "euclidean",
                  cluster_selection_method: str = "eom") -> HDBSCAN:
    """
    Build HDBSCAN model for clustering.
    """
    logger.info("Building HDBSCAN model...")
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True
    )
    return hdbscan_model

