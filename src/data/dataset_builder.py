"""
Dataset builder for Lenta.ru news corpus.

Responsibilities:
- Download dataset from KaggleHub (if not present)
- Load records using Corus
- Convert records to DataFrame
- Save raw and processed versions
"""

import os
import logging
import pandas as pd
import kagglehub
from corus import load_lenta


logger = logging.getLogger(__name__)


def download_lenta_raw(raw_dir: str) -> str:
    """
    Download Lenta.ru dataset using KaggleHub.
    Returns path to the downloaded .csv.gz file.
    """
    os.makedirs(raw_dir, exist_ok=True)

    logger.info("Downloading Lenta.ru dataset from KaggleHub...")
    dataset_path = kagglehub.dataset_download(
        "yutkin/corpus-of-russian-news-articles-from-lenta"
    )

    csv_src = os.path.join(dataset_path, "lenta-ru-news.csv")
    csv_dst = os.path.join(raw_dir, "lenta-ru-news.csv")
    gz_dst = os.path.join(raw_dir, "lenta-ru-news.csv.gz")

    logger.info("Copying CSV to project directory...")
    os.system(f'cp "{csv_src}" "{csv_dst}"')

    logger.info("Compressing CSV to .gz...")
    os.system(f'gzip -kf "{csv_dst}"')

    return gz_dst


def load_raw_records(gz_path: str):
    """
    Load Lenta.ru records using Corus loader.
    """
    logger.info(f"Loading records from {gz_path}...")
    return load_lenta(gz_path)


def build_dataframe(records) -> pd.DataFrame:
    """
    Convert Corus records into a pandas DataFrame.
    """
    logger.info("Converting records to DataFrame...")

    data = []
    for rec in records:
        data.append({
            "title": rec.title,
            "text": rec.text,
            "topic": rec.topic,
        })

    df = pd.DataFrame(data)
    logger.info(f"DataFrame created: {len(df)} rows")
    return df


def save_dataframe(df: pd.DataFrame, path: str):
    """
    Save DataFrame to CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved DataFrame to {path}")

