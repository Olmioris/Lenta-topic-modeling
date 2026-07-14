"""
UMass Coherence metric for BERTopic.

Notes:
- UMass often returns NaN for BERTopic due to sparse c-TF-IDF topics.
- This module handles subsampling and dictionary filtering.
"""

import logging
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

logger = logging.getLogger(__name__)


def umass_coherence(topic_model,
                    texts: list,
                    sample_size: int = 2000,
                    top_k: int = 10,
                    min_freq: int = 5) -> float:
    """
    Compute UMass Coherence for BERTopic model.

    Parameters
    ----------
    topic_model : BERTopic
        Trained BERTopic model.
    texts : list
        List of raw text documents.
    sample_size : int
        Number of documents to use for coherence calculation.
    top_k : int
        Number of top words per topic.
    min_freq : int
        Minimum frequency for dictionary filtering.

    Returns
    -------
    float
        UMass coherence score (may be NaN).
    """
    logger.info("Computing UMass Coherence...")

    # Subsample
    texts_sample = texts[:sample_size]
    tokenized = [t.split() for t in texts_sample]

    # Dictionary
    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=min_freq)

    corpus = [dictionary.doc2bow(t) for t in tokenized]

    # Extract top words per topic
    topics_words = []
    for topic_id in topic_model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(topic_id)[:top_k]]
        topics_words.append(words)

    cm = CoherenceModel(
        topics=topics_words,
        corpus=corpus,
        dictionary=dictionary,
        coherence="u_mass"
    )

    score = cm.get_coherence()
    logger.info(f"UMass Coherence = {score}")
    return score

