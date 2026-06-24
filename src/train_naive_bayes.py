from pathlib import Path
import json
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "naive_bayes"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


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


def load_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    """
    Loads a processed CSV dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} file not found: {path}\n"
            "Please run src/preprocessing.py first."
        )

    df = pd.read_csv(path)

    required_columns = ["text", "label"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column in {dataset_name}: {column}")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, validation and test datasets.
    """
    train_df = load_dataset(TRAIN_DATA_PATH, "Train")
    validation_df = load_dataset(VALIDATION_DATA_PATH, "Validation")
    test_df = load_dataset(TEST_DATA_PATH, "Test")

    return train_df, validation_df, test_df


def print_dataset_info(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """
    Prints basic dataset information.
    """
    print("\nDataset shapes")
    print("-" * 40)
    print(f"Train:      {train_df.shape}")
    print(f"Validation: {validation_df.shape}")
    print(f"Test:       {test_df.shape}")

    print("\nTraining class distribution:")
    print(train_df["label"].map(LABEL_MAP).value_counts())

    print("\nValidation class distribution:")
    print(validation_df["label"].map(LABEL_MAP).value_counts())

    print("\nTest class distribution:")
    print(test_df["label"].map(LABEL_MAP).value_counts())


def build_model() -> Pipeline:
    """
    Builds a TF-IDF + Multinomial Naive Bayes pipeline.
    """
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=30000,
                    min_df=2
                )
            ),
            (
                "classifier",
                MultinomialNB(alpha=1.0)
            )
        ]
    )

    return model


def evaluate_model(
    model: Pipeline,
    df: pd.DataFrame,
    split_name: str
) -> dict:
    """
    Evaluates the model on a given split and saves the report.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x = df["text"]
    y_true = df["label"]

    y_pred = model.predict(x)

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
        "split": split_name,
        "accuracy": accuracy,
        "classification_report": report_dict
    }

    metrics_path = REPORTS_DIR / f"naive_bayes_{split_name}_metrics.json"
    report_path = REPORTS_DIR / f"naive_bayes_{split_name}_classification_report.txt"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    print(f"\n{split_name.capitalize()} accuracy:")
    print(round(accuracy, 4))

    print(f"\n{split_name.capitalize()} classification report:")
    print(report_text)

    print(f"{split_name.capitalize()} metrics saved to: {metrics_path}")
    print(f"{split_name.capitalize()} report saved to: {report_path}")

    return metrics


def save_confusion_matrix(
    model: Pipeline,
    test_df: pd.DataFrame
) -> None:
    """
    Saves the test confusion matrix as an image.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    y_true = test_df["label"]
    y_pred = model.predict(test_df["text"])

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABEL_NAMES
    )

    display.plot(values_format="d")
    plt.title("TF-IDF + Naive Bayes Confusion Matrix")
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_path = FIGURES_DIR / "naive_bayes_confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()

    print(f"\nConfusion matrix saved to: {output_path}")


def save_predictions(
    model: Pipeline,
    test_df: pd.DataFrame
) -> None:
    """
    Saves test predictions for later comparison with DistilBERT.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = test_df.copy()

    predictions = model.predict(test_df["text"])
    probabilities = model.predict_proba(test_df["text"])

    predictions_df["predicted_label"] = predictions
    predictions_df["true_label_name"] = predictions_df["label"].map(LABEL_MAP)
    predictions_df["predicted_label_name"] = predictions_df["predicted_label"].map(LABEL_MAP)

    predictions_df["prob_hate_speech"] = probabilities[:, 0]
    predictions_df["prob_offensive_language"] = probabilities[:, 1]
    predictions_df["prob_neither"] = probabilities[:, 2]

    output_path = REPORTS_DIR / "naive_bayes_test_predictions.csv"
    predictions_df.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")


def save_model(model: Pipeline) -> None:
    """
    Saves the trained TF-IDF + Naive Bayes pipeline.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "tfidf_naive_bayes.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print(f"\nModel saved to: {model_path}")


def save_training_config() -> None:
    """
    Saves baseline model configuration.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "TF-IDF + Multinomial Naive Bayes",
        "tfidf": {
            "lowercase": True,
            "stop_words": "english",
            "ngram_range": [1, 2],
            "max_features": 30000,
            "min_df": 2
        },
        "naive_bayes": {
            "alpha": 1.0
        }
    }

    config_path = REPORTS_DIR / "naive_bayes_training_config.json"

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print(f"Training config saved to: {config_path}")


def main() -> None:
    train_df, validation_df, test_df = load_data()

    print_dataset_info(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df
    )

    model = build_model()

    print("\nTraining TF-IDF + Naive Bayes model...")
    model.fit(train_df["text"], train_df["label"])

    print("\nEvaluating on validation set...")
    evaluate_model(
        model=model,
        df=validation_df,
        split_name="validation"
    )

    print("\nEvaluating on test set...")
    evaluate_model(
        model=model,
        df=test_df,
        split_name="test"
    )

    save_confusion_matrix(model, test_df)
    save_predictions(model, test_df)
    save_model(model)
    save_training_config()


if __name__ == "__main__":
    main()