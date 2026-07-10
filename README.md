# Social Media Analysis for EngD Work Package 1

This repository contains the implementation and analysis code for Work Package 1 of an Engineering Doctorate (EngD) research project focused on social media analysis. The project encompasses sentiment analysis, intent detection, discourse analysis, and relatedness assessment of social media content.

## Project Overview

This research project analyzes social media data through multiple dimensions:

- **Sentiment Detection**: Classifying emotional tone of social media posts
- **Intent Recognition**: Identifying user intentions and motivations
- **Discourse Analysis**: Understanding conversation patterns and themes over time
- **Relatedness Assessment**: Measuring similarity between posts and content
- **Discourse Pattern Discovery**: Identifying emergent discourse types from the joint distribution of sentiment, intent, and relatedness using clustering
- **MLT (Multinomial Logistic Regression Analysis)**: Statistically validating discourse pattern membership against post-level and housing association features

## Repository Structure

```
├── Combine sentiment intent/     # Integration of sentiment and intent analysis
├── data_utils/                   # Data preprocessing and utility functions
├── discourse/                    # Discourse analysis and temporal modeling
├── intent_detection/            # Intent classification models and training
├── intent_utils/                # Intent preprocessing utilities  
├── MLT/                          # Multinomial logistic regression analysis of discourse types
├── relatedness/                 # Content similarity and relatedness analysis
├── sentiment_detection/         # Sentiment analysis models and evaluation
└── requirements.txt             # Python dependencies
```

### Key Components
#### Intent Detection
- **Fine-tuning**: `intent_detection/train.py --model {bert_dutch,deberta,roberta}`
- **Models**: BERT (Dutch), DeBERTa, RoBERTa fine-tuned for intent classification
- **Synthetic Data**: Generated training data for improved model performance
- **Prediction & Visualization**: `intent_pred_and_plot.py`
#### Sentiment Detection  
- **Model Evaluation**: `models_evaluation_sentiment.py`
- **Prediction & Plotting**: `sentiment_prediction_and_plot.py`
#### Discourse Analysis
- **Temporal Analysis**: `discourse_over_time.py` - tracks discourse patterns across time periods (Q1 2018 - Q1 2023)
- **Feature Analysis**: Multiple notebooks analyzing post features and discourse relationships
- **Clustering**: K-means clustering analysis of comment probabilities
- **Multinomial Logistic Modeling**: Advanced statistical modeling of discourse types
### Discourse Pattern Discovery
A core contribution of WP1 is the unsupervised discovery of **six emergent discourse patterns** from the joint distribution of sentiment, intent, and relatedness scores across tenant comments. Key findings:
- **Clustering**: K-means clustering over the combined sentiment, intent, and relatedness feature space reveals six distinct discourse patterns.
- **Characterisation**: Each discourse pattern is profiled by its distinguishing linguistic and structural properties, including sentiment polarity, dominant intent type, and relatedness to the original housing association post
- **Predictive Validation**: Multinomial logistic regression is used to verify that discourse pattern membership is systematically associated with post-level features (e.g., post length, presence of URLs, question framing) and housing association characteristics (e.g., association size, geographic region, portfolio type)
- This validates that the emergent patterns are not arbitrary clusters but reflect meaningful variation in how tenants engage with different types of institutional communication
#### MLT: Multinomial Logistic Regression Analysis
- **Notebook**: `MLT/combined_content_sharing_ref/combined_contentsharing.ipynb`
- **Purpose**: Fits a combined multinomial logit model (`statsmodels.api.MNLogit`) to test whether discourse-cluster membership is predicted by post-level and housing-association (HA) features, with **Content_Sharing** set as the reference discourse category
- **Predictors**:
  - Post features — length category (`Very_Short` / `Short` / `Medium_Long`), unique-word ratio (`Low` / `Medium_High`), presence of questions, presence of URLs
  - HA features — total housing stock, affordability (rent/month), tenant satisfaction (derived from tenants-left, inverted so higher = better) — each discretized into `Low` / `Medium_High` levels
- **Outputs** (written to `MLT/combined_content_sharing_ref/`):
  - `full_results_with_constants.csv` — all coefficients, standard errors, p-values, and intercepts for every discourse type vs. the reference class
  - `significant_results.csv` — predictors significant at p < 0.05
  - `summary.txt` — model fit statistics (pseudo R², AIC, BIC, log-likelihood, convergence)
  - `forest_plot_significant.png`, `forest_plot.png`, `dot_matrix.png`, `combined_plots.png` — coefficient forest plots and dot-matrix visualizations of significant predictors, colour-coded by discourse type
- **Interpretation**: Significant coefficients indicate that features such as HA size, affordability, and tenant satisfaction are systematically associated with specific discourse types (e.g., Information Seeking, On-topic Criticism), providing statistical validation of the discourse patterns discovered via clustering
#### Data Processing
- **Author Extraction**: `extract_author_name.py`
- **Data Cleaning**: `remove_names.py`
- **Preprocessing**: Text normalization, HTML cleaning, URL/mention standardization
#### Relatedness Analysis
- **Similarity Checks**: `Sim_checks_other_posts.py` for cross-post similarity analysis
## Getting Started
### Prerequisites
- Python 3.8+
- Jupyter Notebook/Lab
- CUDA-capable GPU (recommended for model training)
### Installation
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EngD_WP1_Analysis_Social_Media

   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models** (if not automatically downloaded)
   - The scripts will automatically download pre-trained models from Hugging Face
   - Dutch BERT: `GRoNLP/bert-base-dutch-cased`
   - DeBERTa and RoBERTa models as specified in fine-tuning scripts

### Quick Start

1. **Data Preparation**
   - Ensure your data files are in the expected locations
   - Update file paths in the notebooks/scripts to match your data location
   - Run data preprocessing utilities in `data_utils/`

2. **Model Training**
   ```bash
   # Train intent detection models
   python intent_detection/fine_tune_bertje.py
   python intent_detection/fine_tune_deberta.py  
   python intent_detection/fine_tune_robert.py
   ```

3. **Analysis Notebooks**
   - Open Jupyter notebooks in respective directories
   - Run cells sequentially for analysis
   - Modify data paths as needed for your environment

---

**Research Context**: This work is part of an Engineering Doctorate program investigating social media analysis techniques for understanding public discourse and communication patterns.
