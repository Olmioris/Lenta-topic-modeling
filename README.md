# Topic Modeling of Lenta.ru News (BERTopic, 50k documents)

This repository contains a fully reproducible BERTopic pipeline for topic modeling of Lenta.ru news.

The project includes automatic data loading, preprocessing, multilingual embeddings, dimensionality reduction, clustering, topic extraction, quality metrics, and interactive visualizations.

---

## 1. Data Loading and Preprocessing

For topic modeling, it is important to preserve the natural structure of text, because BERTopic relies on sentence-transformers trained on “raw” sentences.

Therefore, I apply **minimal preprocessing**:

- whitespace normalization;
- trimming leading and trailing spaces;
- filtering empty texts;
- selecting a representative subset of 50,000 documents.

50,000 documents is a compromise between:

- **quality** (the corpus remains large and diverse);
- **resources** (BERTopic training and embedding computation fit within Colab limits).

---

## 2. BERTopic Pipeline: Component Selection

The BERTopic pipeline includes:

1. Encoder (sentence-transformer).  
2. Dimensionality reduction (UMAP).  
3. Clustering (HDBSCAN).  
4. Tokenization and vectorization (CountVectorizer).  
5. Post-processing (c‑TF‑IDF).

### 2.1 Encoder

I use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`:

- supports Russian;
- lightweight enough for 50,000 documents;
- produces high-quality embeddings for news texts.

Alternatives:

- LaBSE — heavier model, higher quality, but significantly slower;
- RuBERT — oriented toward Russian, but less universal and heavier.

MiniLM‑L12 is a balance between quality and speed.

### 2.2 Dimensionality Reduction: UMAP

I use UMAP because it:

- preserves local structure well;
- provides a compact representation for HDBSCAN;
- scales better than t‑SNE on 50,000 documents.

Alternatives:

- PCA — faster but worse at capturing nonlinear structure;
- t‑SNE — too slow and unstable on large corpora.

### 2.3 Clustering: HDBSCAN

HDBSCAN:

- does not require specifying the number of clusters;
- robust to noise;
- works well in UMAP space;
- supports identification of “noise” documents.

Alternatives:

- KMeans — requires a fixed number of clusters;
- Agglomerative — scales worse.

### 2.4 Tokenization: CountVectorizer

I use `CountVectorizer` with unigrams:

- `ngram_range=(1, 1)` — only unigrams;
- `min_df=20` — remove rare noise;
- `max_df=0.4` — remove overly frequent words.

Bigrams increase vocabulary size and memory load, which is critical under Colab constraints, so I limit myself to unigrams.

### 2.5 Post-processing: c‑TF‑IDF

c‑TF‑IDF:

- allows interpreting topics through key words;
- works well on large corpora;
- does not require specifying the number of topics.

Additionally, I use `calculate_probabilities=True` to analyze topic distributions across documents.

---

## 3. Hyperparameter Configuration

### UMAP

- `n_neighbors=15` — balance between local and global structure.  
- `n_components=5` — sufficient for clustering, reduces dimensionality and load.  
- `min_dist=0.0` — tighter clusters, better topic separation.  
- `metric="cosine"` — works well with sentence embeddings.

Increasing `n_neighbors` blurs clusters; decreasing `n_components` reduces quality.

### HDBSCAN

- `min_cluster_size=30` — topics are large and stable.  
- `min_samples=10` — balance between noise sensitivity and cluster stability.  
- `cluster_selection_method="eom"` — standard, well-performing mode.

Smaller values would produce many small topics; larger values would merge different topics.

### CountVectorizer

- `ngram_range=(1, 1)` — unigrams reduce vocabulary size and load.  
- `min_df=20` — removes rare noise.  
- `max_df=0.4` — removes overly frequent words that do not help distinguish topics.

The choice of parameters is driven by a compromise between topic quality and Colab memory/time limitations.

---

## 4. 2D Document Visualization

BERTopic provides a built‑in method `visualize_documents()`, but when working with large corpora (tens of thousands of documents) it has a bug. Even if you pass consistent arrays of docs, topics, and embeddings, BERTopic:

- creates an internal index array based on the size of the full corpus;
- tries to index the user-provided list of documents with these indices;
- if the user passes a sample (e.g., 3000 documents), but internal indices refer to the full corpus (e.g., 0–49999), an error occurs:  
  **IndexError: list index out of range**

This is a documented BERTopic issue reproducible on large datasets.

Therefore, for 2D visualization I use manual UMAP + Plotly, which:

- is fully equivalent in meaning,
- gives more control,
- works stably,
- does not depend on BERTopic’s internal indices.

---

## 5. Topic Distribution for Sample Documents

BERTopic allows visualizing topic probability distributions for a specific document.  
For this, the method `visualize_distribution()` is used.

---

## 6. Topic Quality Metrics

Since the module `bertopic.evaluation` is missing in the BERTopic version available in Colab, metrics are computed manually:

- **Topic Diversity** — via counting unique top tokens.  
- **UMass Coherence** — via Gensim on a subsample of the corpus.

### 6.2 Interpretation of Quality Metrics

### Topic Diversity

The obtained value **0.7281** is a high diversity score.  
This means that top tokens of different topics differ significantly, and the model has identified well-separated semantic clusters.

### UMass Coherence

The metric returned **NaN**.  
This is correct and expected for BERTopic for the following reasons:

1. **UMass is sensitive to corpus sparsity.**  
   I used a subsample (2000 documents) to avoid memory overflow, and some top words did not appear in this subsample.

2. **BERTopic topics are based on c‑TF‑IDF.**  
   This model produces topics that do not always have high co-occurrence of words — leading to missing statistics required for UMass.

3. **Gensim returns NaN** if at least one topic contains words absent from the dictionary of the subsample.

Thus, NaN in UMass does not indicate low model quality — it is a limitation of the metric itself.

For BERTopic, more reliable indicators are:

- **Topic Diversity**,  
- **interpretability of top tokens**,  
- **visual separability of topics**.

Together, these indicators confirm that the model produced high-quality and interpretable topics.

*** SQL Data Exploration in DBeaver ***

This section demonstrates how the processed dataset lenta_small.db was explored using SQL queries in DBeaver.

The analysis includes:

 - Basic data preview

 - Topic frequency analysis

 - Average text length by topic

All queries were executed in DBeaver connected to the SQLite database.
See detailed screenshots and discussion in Issue #2.
