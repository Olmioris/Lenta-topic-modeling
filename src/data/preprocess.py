"""
Text preprocessing module.

Responsibilities:
- Clean whitespace
- Filter empty texts
- Sample subset (e.g., 50k)
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def clean_texts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean whitespace and remove empty texts.
    """
    logger.info("Cleaning text fields...")

    df["text"] = (
        df["text"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    logger.info(f"After cleaning: {len(df)} rows")

    return df


def sample_dataset(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    """
    Sample subset of the dataset.
    """
    logger.info(f"Sampling {sample_size} rows...")
    df_small = df.sample(sample_size, random_state=random_state).reset_index(drop=True)
    logger.info(f"Sampled dataset size: {len(df_small)}")
    return df_small

