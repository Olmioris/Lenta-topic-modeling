"""
Embeddings extraction for BERTopic.

Responsibilities:
- Extract document embeddings from trained BERTopic model.
- Save embeddings to .npy file.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


def extract_embeddings(topic_model, texts: list) -> np.ndarray:
    """
    Extract embeddings from BERTopic model.

    Parameters
    ----------
    topic_model : BERTopic
        Trained BERTopic model.
    texts : list
        List of documents.

    Returns
    -------
    np.ndarray
        Embeddings matrix (N x D).
    """
    logger.info("Extracting embeddings from BERTopic model...")
    embeddings = topic_model._extract_embeddings(texts)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings: np.ndarray, path: str):
    """
    Save embeddings to .npy file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)
    logger.info(f"Saved embeddings to {path}")

