import argparse
import yaml
import os
import numpy as np
import pandas as pd

from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def topic_diversity(topic_model, top_k: int):
    topics = topic_model.get_topics()
    unique_words = set()
    total_words = 0

    for topic_id, words in topics.items():
        if topic_id == -1:
            continue
        top_words = [w for w, _ in words[:top_k]]
        unique_words.update(top_words)
        total_words += len(top_words)

    return len(unique_words) / total_words


def compute_umass(topic_model, texts, sample_size: int, top_k: int, min_freq: int):
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

    return cm.get_coherence()


def save_report(report_path: str, diversity: float, umass: float):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# Topic Modeling Evaluation Report\n\n")
        f.write(f"**Topic Diversity:** {diversity:.4f}\n\n")
        f.write(f"**UMass Coherence:** {umass}\n\n")
        f.write("Notes:\n")
        f.write("- Topic Diversity reflects how distinct the topics are.\n")
        f.write("- UMass Coherence may return NaN for BERTopic due to c-TF-IDF sparsity.\n")

    print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BERTopic model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/bertopic_model",
        help="Path to saved BERTopic model"
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="models/embeddings.npy",
        help="Path to saved embeddings"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/lenta_50k_clean.csv",
        help="Path to processed dataset"
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        default="configs/eval.yaml",
        help="Path to evaluation configuration YAML"
    )
    args = parser.parse_args()

    # Load config
    eval_cfg = load_yaml(args.eval_config)

    # Load model
    print("Loading BERTopic model...")
    topic_model = BERTopic.load(args.model_path)

    # Load data
    df = pd.read_csv(args.data_path)
    texts = df["text"].tolist()

    # Topic Diversity
    print("Computing Topic Diversity...")
    diversity = topic_diversity(
        topic_model,
        top_k=eval_cfg["topic_diversity"]["top_k"]
    )

    # UMass Coherence
    print("Computing UMass Coherence...")
    umass = compute_umass(
        topic_model,
        texts,
        sample_size=eval_cfg["umass_coherence"]["sample_size"],
        top_k=eval_cfg["umass_coherence"]["top_k"],
        min_freq=eval_cfg["umass_coherence"]["dictionary_min_freq"]
    )

    # Save report
    if eval_cfg["report"]["save_report"]:
        save_report(
            eval_cfg["report"]["report_path"],
            diversity,
            umass
        )

    print("Evaluation completed.")
    print(f"Topic Diversity: {diversity:.4f}")
    print(f"UMass Coherence: {umass}")


if __name__ == "__main__":
    main()
