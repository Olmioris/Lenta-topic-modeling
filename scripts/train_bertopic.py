import argparse
import yaml
import os

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_and_preprocess_data(data_cfg: dict) -> pd.DataFrame:
    raw_path = data_cfg["dataset"]["download_path"]
    processed_path = data_cfg["dataset"]["processed_path"]
    sample_size = data_cfg["dataset"]["sample_size"]
    random_state = data_cfg["dataset"]["random_state"]

    
    df = pd.read_csv(processed_path)

    
    if data_cfg["preprocessing"]["remove_extra_spaces"]:
        df["text"] = (
            df["text"]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    df = df[df["text"].str.len() >= data_cfg["preprocessing"]["min_text_length"]]
    df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)

    return df


def build_bertopic_model(model_cfg: dict) -> BERTopic:
    # Vectorizer
    vectorizer = CountVectorizer(**model_cfg["vectorizer"])

    # UMAP
    umap_model = UMAP(**model_cfg["umap"])

    # HDBSCAN
    hdbscan_model = HDBSCAN(**model_cfg["hdbscan"])

    # Embedding model
    embedding_model = model_cfg["embedding"]["model_name"]

    # BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language=model_cfg["bertopic"]["language"],
        calculate_probabilities=model_cfg["bertopic"]["calculate_probabilities"],
        low_memory=model_cfg["bertopic"]["low_memory"],
        nr_topics=model_cfg["bertopic"]["nr_topics"],
    )

    return topic_model


def main():
    parser = argparse.ArgumentParser(description="Train BERTopic on Lenta-ru news")
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data.yaml",
        help="Path to data configuration YAML",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_bertopic.yaml",
        help="Path to BERTopic model configuration YAML",
    )
    args = parser.parse_args()

    # Load configs
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)

    # Load and preprocess data
    df = load_and_preprocess_data(data_cfg)
    texts = df["text"].tolist()

    # Build model
    topic_model = build_bertopic_model(model_cfg)

    # Train model
    topics, probs = topic_model.fit_transform(texts)

    # Save model and embeddings
    output_cfg = model_cfg["output"]
    if output_cfg["save_model"]:
        os.makedirs(output_cfg["model_path"], exist_ok=True)
        topic_model.save(output_cfg["model_path"])

    if output_cfg["save_embeddings"]:
        import numpy as np

        embeddings = topic_model._extract_embeddings(texts)
        os.makedirs(os.path.dirname(output_cfg["embeddings_path"]), exist_ok=True)
        np.save(output_cfg["embeddings_path"], embeddings)

    print("Training completed.")
    print(f"Number of topics: {len(set(topics))}")


if __name__ == "__main__":
    main()
