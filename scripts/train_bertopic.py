import argparse
import yaml
import os
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

from src.data.download_lenta import download_lenta_kaggle, load_lenta_records


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_and_preprocess_data(data_cfg: dict) -> pd.DataFrame:
    raw_path = data_cfg["dataset"]["download_path"]
    processed_path = data_cfg["dataset"]["processed_path"]
    sample_size = data_cfg["dataset"]["sample_size"]
    random_state = data_cfg["dataset"]["random_state"]

    # 1. Download raw data if not exists
    if not os.path.exists(raw_path):
        print("Downloading Lenta.ru dataset from Kaggle...")
        gz_path = download_lenta_kaggle(raw_path)
    else:
        gz_path = raw_path

    # 2. Load records using Corus
    print("Loading records...")
    records = load_lenta_records(gz_path)

    # 3. Convert to DataFrame
    data = []
    for record in records:
        data.append({
            "title": record.title,
            "text": record.text,
            "topic": record.topic,
        })

    df = pd.DataFrame(data)

    # 4. Preprocessing
    if data_cfg["preprocessing"]["remove_extra_spaces"]:
        df["text"] = (
            df["text"]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    df = df[df["text"].str.len() >= data_cfg["preprocessing"]["min_text_length"]]

    # 5. Sampling
    df = df.sample(sample_size, random_state=random_state).reset_index(drop=True)

    # 6. Save processed CSV
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)

    print(f"Processed dataset saved to {processed_path}")
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
    print("Training BERTopic model...")
    topics, probs = topic_model.fit_transform(texts)

    # Save model and embeddings
    output_cfg = model_cfg["output"]

    if output_cfg["save_model"]:
        os.makedirs(output_cfg["model_path"], exist_ok=True)
        topic_model.save(output_cfg["model_path"])
        print(f"Model saved to {output_cfg['model_path']}")

    if output_cfg["save_embeddings"]:
        import numpy as np
        embeddings = topic_model._extract_embeddings(texts)
        os.makedirs(os.path.dirname(output_cfg["embeddings_path"]), exist_ok=True)
        np.save(output_cfg["embeddings_path"], embeddings)
        print(f"Embeddings saved to {output_cfg['embeddings_path']}")

    print("Training completed.")
    print(f"Number of topics discovered: {len(set(topics))}")


if __name__ == "__main__":
    main()
