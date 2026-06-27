# Hate Speech Detector

A text mining project for automatic detection of hate speech and offensive language in short social media texts.

The project uses the **Davidson Hate Speech and Offensive Language Dataset** and compares a classical machine learning baseline with a fine-tuned transformer model.

## Project Overview

The goal of this project is to classify tweets into one of three categories:

- `hate_speech` — text containing hate speech directed toward a group of people
- `offensive_language` — offensive or vulgar language that is not necessarily hate speech
- `neither` — text that does not belong to the previous two categories

The main challenge is the distinction between **hate speech** and **offensive language**, because both classes may contain toxic or aggressive vocabulary, but hate speech is specifically related to hostility toward a group identity.

## Dataset

The project uses the **Davidson Hate Speech and Offensive Language Dataset**, which contains tweets manually annotated into three classes:

| Label | Class |
|---|---|
| 0 | hate_speech |
| 1 | offensive_language |
| 2 | neither |

The original dataset contains **24,783 examples**.

The class distribution is highly imbalanced:

| Class | Number of examples |
|---|---:|
| offensive_language | 19,190 |
| neither | 4,163 |
| hate_speech | 1,430 |

Because of this imbalance, the DistilBERT model uses a weighted loss function.

## Project Structure

```text
hate-speech-detector/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── distilbert/
│   └── naive_bayes/
│
├── reports/
│   ├── figures/
│   └── error_report/
│
├── src/
│   ├── download_data.py
│   ├── data_exploration.py
│   ├── preprocessing.py
│   ├── train_distilbert.py
│   ├── evaluate_distilbert.py
│   ├── error_report.py
│   └── train_naive_bayes.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies and Libraries

- **Python** — main programming language
- **Pandas** — data loading, processing and analysis
- **NumPy** — numerical operations
- **Scikit-learn** — TF-IDF, Naive Bayes baseline and evaluation metrics
- **Hugging Face Transformers** — DistilBERT tokenizer and model
- **PyTorch** — model training and weighted loss function
- **Matplotlib / Seaborn** — visualizations and confusion matrices
- **Joblib** — saving the Naive Bayes pipeline
- **Git / GitHub** — version control and project hosting

## Models

Two models are implemented and compared.

### 1. TF-IDF + Multinomial Naive Bayes

This model is used as a classical machine learning baseline.

Text is transformed into numerical features using TF-IDF vectorization with unigrams and bigrams.  
The classifier is `MultinomialNB`.

Main settings:

```text
Vectorizer: TF-IDF
N-grams: unigrams + bigrams
Max features: 30,000
min_df: 2
Classifier: MultinomialNB
Alpha: 1.0
Class weighting: No
```

### 2. Fine-tuned DistilBERT

The main model is based on `distilbert-base-uncased`, fine-tuned for three-class text classification.

Main settings:

```text
Pretrained model: distilbert-base-uncased
Max sequence length: 128
Batch size: 8
Learning rate: 2e-5
Weight decay: 0.01
Epochs: 1 and 2
Loss function: Weighted CrossEntropyLoss
Main metric: Macro F1-score
```

Because the dataset is imbalanced, class weights are used during training:

| Class | Weight |
|---|---:|
| hate_speech | 5.7769 |
| offensive_language | 0.4305 |
| neither | 1.9844 |

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/hate-speech-detector.git
cd hate-speech-detector
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download the dataset

```bash
python src/download_data.py
```

The dataset will be saved in:

```text
data/raw/
```

### 2. Explore the dataset

```bash
python src/data_exploration.py
```

This script performs basic exploratory data analysis and saves figures in:

```text
reports/figures/
```

### 3. Preprocess the data

```bash
python src/preprocessing.py
```

This creates:

```text
data/processed/full_cleaned.csv
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

The preprocessing is intentionally light, because DistilBERT benefits from contextual and structural information in the text.

The following transformations are applied:

- removal of the `RT` marker
- replacement of URLs with a common token
- replacement of user mentions with a common token
- removal of the `#` symbol while keeping hashtag text
- whitespace normalization

### 4. Train the Naive Bayes baseline

```bash
python src/train_naive_bayes.py
```

The trained baseline model is saved in:

```text
models/naive_bayes/
```

Reports and predictions are saved in:

```text
reports/
```

### 5. Train DistilBERT

```bash
python src/train_distilbert.py
```

The fine-tuned model is saved in:

```text
models/distilbert/
```

### 6. Evaluate DistilBERT

```bash
python src/evaluate_distilbert.py
```

This generates:

```text
reports/distilbert_test_metrics.json
reports/distilbert_classification_report.txt
reports/distilbert_test_predictions.csv
reports/figures/distilbert_confusion_matrix.png
```

### 7. Generate error report

```bash
python src/error_report.py
```

The error report is saved in:

```text
reports/error_report/
```

The error report analyzes the most common misclassification patterns for the DistilBERT model.

## Results

The main results on the test set are:

| Model | Accuracy | Macro F1 | Hate Speech F1 | Offensive F1 | Neither F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Naive Bayes | 0.82 | 0.45 | 0.00 | 0.90 | 0.46 |
| DistilBERT — 1 epoch | 0.9112 | 0.77 | 0.47 | 0.95 | 0.90 |
| DistilBERT — 2 epochs | 0.9139 | 0.78 | 0.48 | 0.95 | 0.89 |

The results show that the classical Naive Bayes baseline achieves acceptable accuracy, but fails to detect the `hate_speech` class.

DistilBERT performs significantly better, especially on the minority class, because it uses contextual language representations and weighted loss.

The final selected model is:

```text
DistilBERT — 2 epochs
```

It achieves the best overall performance among the tested models.

## Error Analysis

The error report shows that the most common mistakes are between:

```text
hate_speech
offensive_language
```

This is expected because both classes may contain aggressive, vulgar or toxic language.

The main difficulty is determining whether a text is only offensive or whether it expresses hate toward a group of people.

For the final DistilBERT model with 2 epochs:

```text
Correct predictions: 3398
Wrong predictions: 320
Accuracy: 0.9139
```

The most common error is:

```text
hate_speech predicted as offensive_language: 99 examples
```

This shows that the model sometimes recognizes that a text is offensive, but does not always identify it as hate speech.

## Main Conclusion

The project demonstrates that transformer-based models are more suitable for hate speech detection than classical TF-IDF based models.

The Naive Bayes baseline is useful for comparison, but it is not sufficient for reliable hate speech classification in an imbalanced dataset.

DistilBERT achieves better overall performance and is able to detect part of the minority `hate_speech` class.

However, the task remains challenging because the boundary between hate speech and offensive language is often subtle and context-dependent.

## Future Work

Possible improvements include:

- using a more balanced dataset
- applying oversampling or data augmentation for the `hate_speech` class
- experimenting with larger transformer models
- testing models trained specifically on social media text
- using focal loss or other imbalance-aware loss functions
- performing deeper manual error analysis
- adding explainability methods such as LIME or SHAP
- evaluating the model on an external dataset

## License

This project is created for educational purposes as part of a text mining / natural language processing task.
