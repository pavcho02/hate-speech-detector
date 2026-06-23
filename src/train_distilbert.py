from pathlib import Path
import json
import inspect

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"

MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "distilbert"
RESULTS_DIR = PROJECT_ROOT / "reports"

MODEL_NAME = "distilbert-base-uncased"

NUM_LABELS = 3
MAX_LENGTH = 128

# Full training mode
USE_SMALL_SAMPLE = False
TRAIN_SAMPLE_SIZE = 2000
VALIDATION_SAMPLE_SIZE = 500

NUM_TRAIN_EPOCHS = 1
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

RANDOM_STATE = 42


LABEL_MAP = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither"
}

ID2LABEL = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither"
}

LABEL2ID = {
    "hate_speech": 0,
    "offensive_language": 1,
    "neither": 2
}


class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that uses class weights in the loss function.

    This helps with imbalanced datasets where one class has far fewer examples.
    In our dataset, hate_speech is much smaller than offensive_language.
    """

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_function = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_function = torch.nn.CrossEntropyLoss()

        loss = loss_function(
            logits.view(-1, NUM_LABELS),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train and validation data created by preprocessing.py.
    """
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Train file not found: {TRAIN_DATA_PATH}\n"
            "Please run src/preprocessing.py first."
        )

    if not VALIDATION_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_DATA_PATH}\n"
            "Please run src/preprocessing.py first."
        )

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    validation_df = pd.read_csv(VALIDATION_DATA_PATH)

    return train_df, validation_df


def sample_dataframe(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Creates a stratified sample from the dataframe.

    This is useful for quick testing because DistilBERT training can be slow
    on CPU.
    """
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)

    sampled_df, _ = train_test_split(
        df,
        train_size=sample_size,
        random_state=random_state,
        stratify=df["label"]
    )

    return sampled_df.reset_index(drop=True)


def prepare_dataframes(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keeps only the columns needed for model training.
    """
    required_columns = ["text", "label"]

    for column in required_columns:
        if column not in train_df.columns:
            raise ValueError(f"Missing column in train data: {column}")

        if column not in validation_df.columns:
            raise ValueError(f"Missing column in validation data: {column}")

    train_df = train_df[["text", "label"]].copy()
    validation_df = validation_df[["text", "label"]].copy()

    train_df["label"] = train_df["label"].astype(int)
    validation_df["label"] = validation_df["label"].astype(int)

    if USE_SMALL_SAMPLE:
        train_df = sample_dataframe(train_df, TRAIN_SAMPLE_SIZE)
        validation_df = sample_dataframe(validation_df, VALIDATION_SAMPLE_SIZE)

    return train_df, validation_df


def calculate_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Calculates class weights based on the training set distribution.

    Smaller classes receive larger weights.
    """
    class_labels = np.array([0, 1, 2])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_labels,
        y=train_df["label"].to_numpy()
    )

    class_weights = torch.tensor(weights, dtype=torch.float)

    print("\nClass weights:")
    for label_id, weight in zip(class_labels, weights):
        print(f"{LABEL_MAP[label_id]}: {weight:.4f}")

    return class_weights


def convert_to_huggingface_dataset(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame
) -> tuple[Dataset, Dataset]:
    """
    Converts pandas DataFrames to Hugging Face Dataset objects.
    """
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    validation_dataset = Dataset.from_pandas(validation_df, preserve_index=False)

    return train_dataset, validation_dataset


def tokenize_datasets(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    tokenizer
) -> tuple[Dataset, Dataset]:
    """
    Tokenizes the text column so it can be used by DistilBERT.
    """
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    tokenized_train = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"]
    )

    tokenized_validation = validation_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"]
    )

    return tokenized_train, tokenized_validation


def compute_metrics(eval_pred):
    """
    Calculates evaluation metrics during validation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


def build_training_arguments() -> TrainingArguments:
    """
    Creates TrainingArguments.

    Some versions of transformers use 'eval_strategy',
    older ones use 'evaluation_strategy'. This function supports both.
    """
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args_kwargs = {
        "output_dir": str(MODEL_OUTPUT_DIR),
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 100,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": "none",
        "seed": RANDOM_STATE
    }

    training_args_parameters = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    if "eval_strategy" in training_args_parameters:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch"

    if "dataloader_pin_memory" in training_args_parameters:
        training_args_kwargs["dataloader_pin_memory"] = torch.cuda.is_available()

    return TrainingArguments(**training_args_kwargs)


def build_trainer(
    model,
    training_args: TrainingArguments,
    tokenized_train: Dataset,
    tokenized_validation: Dataset,
    tokenizer,
    class_weights: torch.Tensor
) -> Trainer:
    """
    Creates a WeightedLossTrainer instance.

    Different versions of transformers use different parameter names:
    - newer versions use processing_class
    - older versions use tokenizer

    This function supports both.
    """
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_validation,
        "compute_metrics": compute_metrics,
        "class_weights": class_weights
    }

    trainer_parameters = inspect.signature(Trainer.__init__).parameters

    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    return WeightedLossTrainer(**trainer_kwargs)


def print_device_info() -> None:
    """
    Prints information about whether training will use CPU or GPU.
    """
    print("\nDevice information")
    print("-" * 40)

    if torch.cuda.is_available():
        print("CUDA is available. Training will use GPU.")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training will use CPU.")
        print("Local CPU training may be slow.")


def save_validation_metrics(metrics: dict) -> None:
    """
    Saves validation metrics to reports/distilbert_validation_metrics.json.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = RESULTS_DIR / "distilbert_validation_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nValidation metrics saved to: {metrics_path}")


def save_training_config(class_weights: torch.Tensor) -> None:
    """
    Saves training configuration for documentation and reproducibility.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "model_name": MODEL_NAME,
        "num_labels": NUM_LABELS,
        "max_length": MAX_LENGTH,
        "use_small_sample": USE_SMALL_SAMPLE,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "class_weights": {
            LABEL_MAP[index]: float(weight)
            for index, weight in enumerate(class_weights.tolist())
        }
    }

    config_path = RESULTS_DIR / "distilbert_training_config.json"

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print(f"Training config saved to: {config_path}")


def main() -> None:
    print_device_info()

    train_df, validation_df = load_processed_data()
    train_df, validation_df = prepare_dataframes(train_df, validation_df)

    print("\nTraining data shape:", train_df.shape)
    print("Validation data shape:", validation_df.shape)

    print("\nTraining class distribution:")
    print(train_df["label"].map(LABEL_MAP).value_counts())

    print("\nValidation class distribution:")
    print(validation_df["label"].map(LABEL_MAP).value_counts())

    class_weights = calculate_class_weights(train_df)
    save_training_config(class_weights)

    train_dataset, validation_dataset = convert_to_huggingface_dataset(
        train_df,
        validation_df
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_train, tokenized_validation = tokenize_datasets(
        train_dataset,
        validation_dataset,
        tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    training_args = build_training_arguments()

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        tokenized_train=tokenized_train,
        tokenized_validation=tokenized_validation,
        tokenizer=tokenizer,
        class_weights=class_weights
    )

    print("\nStarting full DistilBERT training...")
    trainer.train()

    print("\nEvaluating on validation set...")
    validation_metrics = trainer.evaluate()
    print(validation_metrics)

    save_validation_metrics(validation_metrics)

    print("\nSaving model and tokenizer...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    print(f"\nModel saved to: {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()