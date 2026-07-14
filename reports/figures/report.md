# Topic Modeling Evaluation Report

This report summarizes the evaluation metrics for BERTopic model trained on Lenta.ru (50k documents).

## Metrics

- **Topic Diversity:** (value will be inserted automatically)
- **UMass Coherence:** (value will be inserted automatically)

## Notes

- Topic Diversity reflects how distinct the topics are.
- UMass Coherence may return NaN for BERTopic due to sparse c-TF-IDF topics.
- Visualizations are stored in `reports/figures/`.

## Generated Files

- `docs_2d.html` — UMAP projection of document embeddings
- `topic_tokens.png` — top tokens per topic
- `topic_distributions.html` — distribution of topics across documents
