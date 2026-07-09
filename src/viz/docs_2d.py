import os
import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP


def plot_docs_2d(embeddings: np.ndarray,
                 topics: list,
                 output_dir: str,
                 filename: str = "docs_2d.html",
                 n_neighbors: int = 15,
                 min_dist: float = 0.0,
                 random_state: int = 42):
    """
    Create a 2D UMAP projection of document embeddings and save as an HTML file.

    Parameters
    ----------
    embeddings : np.ndarray
        Document embeddings (N x D).
    topics : list
        Topic assignments for each document.
    output_dir : str
        Directory where the visualization will be saved.
    filename : str
        Name of the output HTML file.
    n_neighbors : int
        UMAP parameter controlling local structure.
    min_dist : float
        UMAP parameter controlling cluster tightness.
    random_state : int
        Random seed for reproducibility.
    """

    # 1. UMAP projection
    umap_2d = UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state
    ).fit_transform(embeddings)

    # 2. Build DataFrame
    df_plot = pd.DataFrame({
        "x": umap_2d[:, 0],
        "y": umap_2d[:, 1],
        "topic": topics
    })

    # 3. Plot
    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="topic",
        title="Documents in 2D space (UMAP)",
        opacity=0.7
    )

    # 4. Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    fig.write_html(output_path)
    print(f"2D UMAP visualization saved to {output_path}")

