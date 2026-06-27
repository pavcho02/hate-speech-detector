# Hate Speech Detection with DistilBERT

This project focuses on automatic hate speech detection in text using **DistilBERT**, a lightweight transformer-based language model. The task is formulated as a supervised text classification problem, where each input text is classified into one of several categories such as **hate speech**, **offensive language**, or **neutral content**.

---

## 📖 Project Overview

The main goal of the project is to build and evaluate a machine learning model capable of identifying hate speech in short text messages, such as tweets or online comments. The project uses **transfer learning** by fine-tuning a pretrained **DistilBERT** model on an annotated hate speech dataset.

---

## 📂 Dataset

The project uses a publicly available dataset from **Hugging Face** or **Kaggle**, such as the **Davidson Hate Speech Dataset**.

### Classes

| Label | Class |
|------:|----------------------|
| 0 | Hate Speech |
| 1 | Offensive Language |
| 2 | Neutral / Neither |

The dataset is imbalanced, therefore multiple evaluation metrics are used instead of relying solely on accuracy.

---

## ⚙️ Methodology

The implementation follows these main steps:

1. Load the dataset
2. Analyze the class distribution
3. Clean and preprocess the text
4. Split the data into training, validation and test sets
5. Tokenize the text using the DistilBERT tokenizer
6. Fine-tune the DistilBERT model
7. Evaluate the model
8. Analyze the results

---

## 📝 Text Preprocessing

Minimal preprocessing is applied because DistilBERT relies on contextual information.

Applied preprocessing includes:

- Removing empty records
- Removing URLs
- Removing user mentions
- Cleaning HTML entities
- Removing duplicated samples
- Normalizing whitespace

**Not applied:**

- Stop-word removal
- Stemming
- Lemmatization

---

## 🤖 Model

The project uses:

```python
DistilBertForSequenceClassification
```

DistilBERT is selected because it provides an excellent balance between accuracy and computational efficiency while preserving most of BERT's language understanding capabilities.

---

## 🎯 Training

The model is trained using **supervised learning**.

Training pipeline:

- Tokenize text
- Generate contextual embeddings
- Compute predictions
- Calculate CrossEntropy Loss
- Update model weights using backpropagation
- Optimize parameters with AdamW

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Because hate speech datasets are highly imbalanced, F1-score is considered one of the most important evaluation metrics.

---

## 🛠 Technologies

- Python
- Pandas
- NumPy
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab

---

## 📦 Installation

```bash
pip install pandas numpy torch transformers datasets scikit-learn matplotlib seaborn
```

---

## ▶️ Running the Project

Clone the repository:

```bash
git clone https://github.com/your-username/hate-speech-detection-distilbert.git
cd hate-speech-detection-distilbert
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python train.py
```

or

```bash
jupyter notebook
```

---

## 📁 Project Structure

```text
hate-speech-detection-distilbert/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── hate_speech_distilbert.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── distilbert_model/
│
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── training_history.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 💬 Example Prediction

**Input**

```text
You are such an idiot.
```

**Prediction**

```text
Offensive Language
```

---

## 🚀 Future Improvements

Possible future work includes:

- Fine-tuning larger transformer models (BERT, RoBERTa)
- Handling class imbalance using weighted loss or oversampling
- Hyperparameter optimization
- Explainable AI techniques (LIME, SHAP)
- Multilingual hate speech detection
- Bulgarian hate speech classification

---

## 📚 References

- Devlin et al. (2018). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
- Sanh et al. (2019). **DistilBERT: A distilled version of BERT**
- Davidson et al. (2017). **Automated Hate Speech Detection and the Problem of Offensive Language**

---

## 📄 License

This project is intended for educational and research purposes.