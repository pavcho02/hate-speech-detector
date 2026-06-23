from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MAX_LENGTH = 128
BATCH_SIZE = 16

LABEL_MAP = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither"
}

LABEL_NAMES = [
    "hate_speech",
    "offensive_language",
    "neither"
]


def load_test_data() -> pd.DataFrame:
    """
    Loads the test dataset created by preprocessing.py.
    """
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test file not found: {TEST_DATA_PATH}\n"
            "Please run src/preprocessing.py first."
        )

    test_df = pd.read_csv(TEST_DATA_PATH)

    required_columns = ["text", "label"]

    for column in required_columns:
        if column not in test_df.columns:
            raise ValueError(f"Missing required column in test data: {column}")

    test_df = test_df[["text", "label"]].copy()
    test_df["label"] = test_df["label"].astype(int)

    return test_df


def load_model_and_tokenizer():
    """
    Loads the fine-tuned DistilBERT model and tokenizer.
    """
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}\n"
            "Please run src/train_distilbert.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    return model, tokenizer


def get_device() -> torch.device:
    """
    Selects GPU if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        print("CUDA is available. Evaluation will use GPU.")
        return torch.device("cuda")

    print("CUDA is not available. Evaluation will use CPU.")
    return torch.device("cpu")


def predict(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs model predictions in batches.

    Returns:
    - predicted class labels
    - class probabilities
    """
    model.to(device)
    model.eval()

    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for start_index in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start_index:start_index + BATCH_SIZE]

            encoded_batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            encoded_batch = {
                key: value.to(device)
                for key, value in encoded_batch.items()
            }

            outputs = model(**encoded_batch)
            logits = outputs.logits

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_probabilities)


def save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> None:
    """
    Calculates and saves evaluation metrics.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    accuracy = accuracy_score(y_true, y_pred)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=LABEL_NAMES,
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "classification_report": report_dict
    }

    metrics_path = REPORTS_DIR / "distilbert_test_metrics.json"
    report_path = REPORTS_DIR / "distilbert_classification_report.txt"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    print("\nTest accuracy:")
    print(round(accuracy, 4))

    print("\nClassification report:")
    print(report_text)

    print(f"Metrics saved to: {metrics_path}")
    print(f"Text report saved to: {report_path}")


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> None:
    """
    Saves the confusion matrix as an image.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABEL_NAMES
    )

    display.plot(values_format="d")
    plt.title("DistilBERT Confusion Matrix")
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_path = FIGURES_DIR / "distilbert_confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")


def save_predictions(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray
) -> None:
    """
    Saves model predictions for later error analysis.
    """
    predictions_df = test_df.copy()

    predictions_df["predicted_label"] = predictions
    predictions_df["true_label_name"] = predictions_df["label"].map(LABEL_MAP)
    predictions_df["predicted_label_name"] = predictions_df["predicted_label"].map(LABEL_MAP)

    predictions_df["prob_hate_speech"] = probabilities[:, 0]
    predictions_df["prob_offensive_language"] = probabilities[:, 1]
    predictions_df["prob_neither"] = probabilities[:, 2]

    output_path = REPORTS_DIR / "distilbert_test_predictions.csv"
    predictions_df.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")


def main() -> None:
    test_df = load_test_data()

    print("Test data shape:")
    print(test_df.shape)

    print("\nTest class distribution:")
    print(test_df["label"].map(LABEL_MAP).value_counts())

    model, tokenizer = load_model_and_tokenizer()
    device = get_device()

    texts = test_df["text"].tolist()
    y_true = test_df["label"].to_numpy()

    print("\nRunning predictions on test set...")
    y_pred, probabilities = predict(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device
    )

    save_metrics(y_true, y_pred)
    save_confusion_matrix(y_true, y_pred)
    save_predictions(test_df, y_pred, probabilities)


if __name__ == "__main__":
    main()