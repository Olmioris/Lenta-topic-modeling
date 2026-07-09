import os
import kagglehub
from corus import load_lenta


def download_lenta_kaggle(download_path: str) -> str:
    """
    Download the Lenta.ru dataset using kagglehub and return the path
    to the downloaded CSV.GZ file.
    """
    dataset_path = kagglehub.dataset_download(
        "yutkin/corpus-of-russian-news-articles-from-lenta"
    )

    src_csv = os.path.join(dataset_path, "lenta-ru-news.csv")
    dst_csv = download_path.replace(".gz", "")
    dst_gz = download_path

    # Copy and gzip
    os.makedirs(os.path.dirname(dst_csv), exist_ok=True)
    os.system(f'cp "{src_csv}" "{dst_csv}"')
    os.system(f'gzip -kf "{dst_csv}"')

    return dst_gz


def load_lenta_records(gz_path: str):
    """
    Load Lenta.ru records using Corus.
    Returns a generator of records.
    """
    return load_lenta(gz_path)
