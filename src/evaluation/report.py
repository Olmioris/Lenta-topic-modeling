"""
Report builder for BERTopic evaluation.

Responsibilities:
- Save Topic Diversity and UMass Coherence into a Markdown report.
"""

import os
import logging

logger = logging.getLogger(__name__)


def save_report(report_path: str,
                diversity: float,
                umass: float):
    """
    Save evaluation metrics into a Markdown report.

    Parameters
    ----------
    report_path : str
        Output path for the report.
    diversity : float
        Topic Diversity score.
    umass : float
        UMass Coherence score (may be NaN).
    """
    logger.info(f"Saving evaluation report to {report_path}...")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# Topic Modeling Evaluation Report\n\n")
        f.write(f"**Topic Diversity:** {diversity:.4f}\n\n")
        f.write(f"**UMass Coherence:** {umass}\n\n")
        f.write("### Notes\n")
        f.write("- Topic Diversity reflects how distinct the topics are.\n")
        f.write("- UMass Coherence may return NaN for BERTopic due to sparse c-TF-IDF topics.\n")
        f.write("- For BERTopic, Topic Diversity and interpretability are more reliable indicators.\n")
    logger.info("Report saved successfully.")

