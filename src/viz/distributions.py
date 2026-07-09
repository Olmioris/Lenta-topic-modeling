import os
import pandas as pd
import plotly.express as px


def plot_topic_distributions(topics: list,
                             output_dir: str,
                             filename: str = "topic_distributions.html"):
    """
    Create a bar chart showing how many documents belong to each topic
    and save it as an interactive HTML file.

    Parameters
    ----------
    topics : list
        Topic assignment for each document.
    output_dir : str
        Directory where the visualization will be saved.
    filename : str
        Name of the output HTML file.
    """

    # Count documents per topic
    df = pd.DataFrame({"topic": topics})
    counts = df["topic"].value_counts().reset_index()
    counts.columns = ["topic", "count"]

    # Remove outlier topic -1 (BERTopic noise)
    counts = counts[counts["topic"] != -1]

    # Plot
    fig = px.bar(
        counts,
        x="topic",
        y="count",
        title="Topic Distribution Across Documents",
        labels={"topic": "Topic ID", "count": "Number of Documents"},
        height=600
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    fig.write_html(output_path)

    print(f"Topic distribution visualization saved to {output_path}")

