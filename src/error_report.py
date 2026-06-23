from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "distilbert_test_predictions.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ERROR_REPORT_DIR = REPORTS_DIR / "error_report"


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


def load_predictions() -> pd.DataFrame:
    """
    Loads the test predictions created by evaluate_distilbert.py.
    """
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}\n"
            "Please run src/evaluate_distilbert.py first."
        )

    predictions_df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = [
        "text",
        "label",
        "predicted_label",
        "true_label_name",
        "predicted_label_name",
        "prob_hate_speech",
        "prob_offensive_language",
        "prob_neither"
    ]

    for column in required_columns:
        if column not in predictions_df.columns:
            raise ValueError(f"Missing required column: {column}")

    predictions_df["label"] = predictions_df["label"].astype(int)
    predictions_df["predicted_label"] = predictions_df["predicted_label"].astype(int)

    return predictions_df


def create_confusion_matrix_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates and saves a confusion matrix as a CSV table.
    """
    y_true = predictions_df["label"]
    y_pred = predictions_df["predicted_label"]

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in LABEL_NAMES],
        columns=[f"predicted_{label}" for label in LABEL_NAMES]
    )

    output_path = ERROR_REPORT_DIR / "confusion_matrix_table.csv"
    matrix_df.to_csv(output_path)

    print("\nConfusion matrix table:")
    print(matrix_df)

    print(f"\nConfusion matrix table saved to: {output_path}")

    return matrix_df


def save_confusion_matrix_plot(predictions_df: pd.DataFrame) -> None:
    """
    Saves a confusion matrix image for the error report.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    y_true = predictions_df["label"]
    y_pred = predictions_df["predicted_label"]

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABEL_NAMES
    )

    display.plot(values_format="d")
    plt.title("DistilBERT Test Confusion Matrix")
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_path = FIGURES_DIR / "distilbert_error_report_confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Confusion matrix plot saved to: {output_path}")


def get_error_pairs(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the most common pairs of true label and predicted label
    for the misclassified examples.
    """
    misclassified_df = predictions_df[
        predictions_df["label"] != predictions_df["predicted_label"]
    ].copy()

    error_pairs = (
        misclassified_df
        .groupby(["true_label_name", "predicted_label_name"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
        .reset_index(drop=True)
    )

    return error_pairs


def save_error_subsets(predictions_df: pd.DataFrame) -> None:
    """
    Saves different subsets of model errors for manual inspection.
    """
    ERROR_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    misclassified_df = predictions_df[
        predictions_df["label"] != predictions_df["predicted_label"]
    ].copy()

    correct_df = predictions_df[
        predictions_df["label"] == predictions_df["predicted_label"]
    ].copy()

    hate_false_negatives = predictions_df[
        (predictions_df["label"] == 0) &
        (predictions_df["predicted_label"] != 0)
    ].copy()

    hate_false_positives = predictions_df[
        (predictions_df["label"] != 0) &
        (predictions_df["predicted_label"] == 0)
    ].copy()

    offensive_as_hate = predictions_df[
        (predictions_df["label"] == 1) &
        (predictions_df["predicted_label"] == 0)
    ].copy()

    hate_as_offensive = predictions_df[
        (predictions_df["label"] == 0) &
        (predictions_df["predicted_label"] == 1)
    ].copy()

    selected_columns = [
        "text",
        "true_label_name",
        "predicted_label_name",
        "prob_hate_speech",
        "prob_offensive_language",
        "prob_neither"
    ]

    files_to_save = {
        "all_misclassified.csv": misclassified_df[selected_columns],
        "all_correct.csv": correct_df[selected_columns],
        "hate_speech_false_negatives.csv": hate_false_negatives[selected_columns],
        "hate_speech_false_positives.csv": hate_false_positives[selected_columns],
        "offensive_language_predicted_as_hate_speech.csv": offensive_as_hate[selected_columns],
        "hate_speech_predicted_as_offensive_language.csv": hate_as_offensive[selected_columns]
    }

    for filename, dataframe in files_to_save.items():
        output_path = ERROR_REPORT_DIR / filename
        dataframe.to_csv(output_path, index=False)

    print("\nError report CSV files saved to:")
    print(ERROR_REPORT_DIR)


def create_error_summary(
    predictions_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    error_pairs: pd.DataFrame
) -> None:
    """
    Creates a factual text summary based only on the model outputs.
    """
    total_examples = len(predictions_df)

    correct_predictions = (
        predictions_df["label"] == predictions_df["predicted_label"]
    ).sum()

    wrong_predictions = total_examples - correct_predictions
    accuracy = correct_predictions / total_examples

    hate_false_negatives = predictions_df[
        (predictions_df["label"] == 0) &
        (predictions_df["predicted_label"] != 0)
    ].copy()

    hate_false_positives = predictions_df[
        (predictions_df["label"] != 0) &
        (predictions_df["predicted_label"] == 0)
    ].copy()

    prediction_distribution = (
        predictions_df["predicted_label_name"]
        .value_counts()
        .to_string()
    )

    true_distribution = (
        predictions_df["true_label_name"]
        .value_counts()
        .to_string()
    )

    summary_lines = []

    summary_lines.append("DistilBERT Error Report")
    summary_lines.append("=" * 40)
    summary_lines.append("")
    summary_lines.append(f"Total test examples: {total_examples}")
    summary_lines.append(f"Correct predictions: {correct_predictions}")
    summary_lines.append(f"Wrong predictions: {wrong_predictions}")
    summary_lines.append(f"Accuracy: {accuracy:.4f}")
    summary_lines.append("")
    summary_lines.append("True label distribution:")
    summary_lines.append(true_distribution)
    summary_lines.append("")
    summary_lines.append("Predicted label distribution:")
    summary_lines.append(prediction_distribution)
    summary_lines.append("")
    summary_lines.append("Confusion matrix:")
    summary_lines.append(matrix_df.to_string())
    summary_lines.append("")
    summary_lines.append("Most common error types:")
    summary_lines.append(error_pairs.to_string(index=False))
    summary_lines.append("")
    summary_lines.append(f"Hate speech false negatives: {len(hate_false_negatives)}")
    summary_lines.append(f"Hate speech false positives: {len(hate_false_positives)}")

    if not error_pairs.empty:
        top_error = error_pairs.iloc[0]
        summary_lines.append("")
        summary_lines.append("Largest error category:")
        summary_lines.append(
            f"{top_error['true_label_name']} predicted as "
            f"{top_error['predicted_label_name']}: {top_error['count']} examples"
        )

    summary_lines.append("")
    summary_lines.append("Note:")
    summary_lines.append(
        "This file reports observed error patterns from the test predictions. "
        "Interpretation of these errors should be done in the project report "
        "based on the confusion matrix, classification report and manual review "
        "of misclassified examples."
    )

    output_path = ERROR_REPORT_DIR / "error_report_summary.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(summary_lines))

    print(f"\nError report summary saved to: {output_path}")


def main() -> None:
    ERROR_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = load_predictions()

    print("Predictions shape:")
    print(predictions_df.shape)

    print("\nPrediction distribution:")
    print(predictions_df["predicted_label_name"].value_counts())

    print("\nTrue label distribution:")
    print(predictions_df["true_label_name"].value_counts())

    matrix_df = create_confusion_matrix_table(predictions_df)
    error_pairs = get_error_pairs(predictions_df)

    save_confusion_matrix_plot(predictions_df)
    save_error_subsets(predictions_df)
    create_error_summary(predictions_df, matrix_df, error_pairs)

    print("\nMost common error types:")
    print(error_pairs)


if __name__ == "__main__":
    main()