"""
BERTopic pipeline.

Responsibilities:
- Build BERTopic model with given components.
- Fit model on texts.
- Save model and embeddings.
"""

import os
import logging
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

from src.modeling.clustering import build_umap, build_hdbscan
from src.modeling.embeddings import extract_embeddings, save_embeddings

logger = logging.getLogger(__name__)


def build_vectorizer(min_df: int = 20,
                     max_df: float = 0.4,
                     ngram_range=(1, 1)) -> CountVectorizer:
    """
    Build CountVectorizer for BERTopic.
    """
    logger.info("Building CountVectorizer...")
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    return vectorizer


def build_bertopic_model(embedding_model: str,
                         random_state: int = 42):
    """
    Build BERTopic model with configured components.
    """
    logger.info("Building BERTopic model...")

    vectorizer_model = build_vectorizer()
    umap_model = build_umap(random_state=random_state)
    hdbscan_model = build_hdbscan()

    topic_model = BERTopic(
        language="multilingual",
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=True,
        low_memory=True,
    )

    return topic_model


def fit_bertopic(topic_model,
                 texts: list):
    """
    Fit BERTopic model on texts.
    """
    logger.info("Fitting BERTopic model...")
    topics, probs = topic_model.fit_transform(texts)
    logger.info("BERTopic training completed.")
    return topics, probs


def save_bertopic_model(topic_model,
                        model_dir: str):
    """
    Save BERTopic model to directory.
    """
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Saving BERTopic model to {model_dir}...")
    topic_model.save(model_dir)
    logger.info("BERTopic model saved.")


def run_bertopic_pipeline(texts: list,
                          embedding_model: str,
                          model_dir: str,
                          embeddings_path: str):
    """
    Full BERTopic pipeline:
    - build model
    - fit on texts
    - save model
    - extract and save embeddings
    """
    topic_model = build_bertopic_model(
        embedding_model=embedding_model,
        random_state=42
    )

    topics, probs = fit_bertopic(topic_model, texts)
    save_bertopic_model(topic_model, model_dir)

    embeddings = extract_embeddings(topic_model, texts)
    save_embeddings(embeddings, embeddings_path)

    return topic_model, topics, probs, embeddings

