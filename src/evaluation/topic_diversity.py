"""
Topic Diversity metric for BERTopic.

Definition:
    Topic Diversity = (# unique words across all topics) / (# total words across all topics)

Used to measure how distinct topics are from each other.
"""

import logging

logger = logging.getLogger(__name__)


def topic_diversity(topic_model, top_k: int = 10) -> float:
    """
    Compute Topic Diversity for BERTopic model.

    Parameters
    ----------
    topic_model : BERTopic
        Trained BERTopic model.
    top_k : int
        Number of top words to consider per topic.

    Returns
    -------
    float
        Topic Diversity score.
    """
    logger.info("Computing Topic Diversity...")

    topics = topic_model.get_topics()
    unique_words = set()
    total_words = 0

    for topic_id, words in topics.items():
        if topic_id == -1:
            continue

        top_words = [w for w, _ in words[:top_k]]
        unique_words.update(top_words)
        total_words += len(top_words)

    diversity = len(unique_words) / total_words if total_words > 0 else 0.0

    logger.info(f"Topic Diversity = {diversity:.4f}")
    return diversity

