# Quickstart Guide

## 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download nl_core_news_lg   # Dutch NER model used by data_utils/remove_names.py
```

## 2. Set your data paths

Every script has a clearly labelled `CONFIGURATION` block at the top.
Edit the path constants to point to your local files before running.

For `intent_detection/train.py` you can pass the path directly as a CLI argument
(no editing needed — see step 3).

## 3. Run the pipeline

### Intent classification — fine-tune a model

```bash
# Fine-tune BERTje (Dutch BERT)
python intent_detection/train.py --model bert_dutch --data path/to/data.xlsx

# Fine-tune DeBERTa
python intent_detection/train.py --model deberta --data path/to/data.xlsx

# Fine-tune RoBERTa
python intent_detection/train.py --model roberta --data path/to/data.xlsx
```

Optional flags: `--seed INT`, `--n-splits INT`, `--max-length INT`

### Intent prediction on new data

```bash
python intent_detection/intent_pred_and_plot.py
```

Update `config.py` (`DEFAULT_DATA_PATHS`, `MODEL_SAVE_PATHS`) to point to your
data and saved models before running.

### Data preprocessing

```bash
python data_utils/remove_names.py         # anonymise person/org names via spaCy NER
python data_utils/extract_author_name.py  # enrich data with author metadata
```

### Sentiment analysis

```bash
python sentiment_detection/sentiment_prediction_and_plot.py   # predict and plot
python sentiment_detection/models_evaluation_sentiment.py     # evaluate candidate models
```

### Discourse analysis

```bash
python discourse/KMeans_cluster_each_comments_probs.py    # K-Means clustering
python discourse/discourse_over_time.py                   # timeline visualisation
```

### MLT — multinomial logistic regression analysis

Statistically validates discourse-cluster membership (from the K-Means step above)
against post-level and housing-association features, with `Content_Sharing` as the
reference class.

```bash
jupyter nbconvert --to notebook --execute --inplace \
  MLT/combined_content_sharing_ref/combined_contentsharing.ipynb
```

Before running, edit the `DATA_PATH` and `OUT_DIR` constants near the top of the
notebook to point to your clustered data file and desired output folder. Outputs
(`full_results_with_constants.csv`, `significant_results.csv`, `summary.txt`, and
forest/dot-matrix plots) are written to `MLT/combined_content_sharing_ref/`.

### Combine sentiment and intent predictions

```bash
python "Combine sentiment intent/combine.py"
```

## 4. Run tests

```bash
pytest tests/ -v
```

## 5. Check your environment

```bash
python config.py
```

Prints Python, PyTorch, and Transformers versions, CUDA availability, and all
configured directory paths.
