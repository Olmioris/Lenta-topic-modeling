import os
import pandas as pd
import plotly.express as px


def plot_topic_tokens(topic_model,
                      output_dir: str,
                      filename: str = "topic_tokens.png",
                      top_n_words: int = 10):
    """
    Create a bar chart of top words for each topic and save as PNG.

    Parameters
    ----------
    topic_model : BERTopic
        Trained BERTopic model.
    output_dir : str
        Directory where the visualization will be saved.
    filename : str
        Name of the output PNG file.
    top_n_words : int
        Number of top words to display per topic.
    """

    topics = topic_model.get_topics()
    rows = []

    for topic_id, words in topics.items():
        if topic_id == -1:
            continue

        for word, score in words[:top_n_words]:
            rows.append({
                "topic": topic_id,
                "word": word,
                "score": score
            })

    df = pd.DataFrame(rows)

    fig = px.bar(
        df,
        x="word",
        y="score",
        color="topic",
        title="Top Words per Topic",
        barmode="group",
        height=800
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    fig.write_image(output_path)
    print(f"Topic tokens visualization saved to {output_path}")

