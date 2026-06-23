from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "labeled_data.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


LABEL_MAP = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither"
}


def clean_tweet(text: str) -> str:
    """
    Applies light preprocessing to a tweet.

    We avoid aggressive cleaning because transformer models such as DistilBERT
    use context, punctuation and subword information.
    """
    text = str(text)

    # Remove retweet marker at the beginning of the tweet
    text = re.sub(r"^RT\s+", "", text)

    # Replace URLs with a common token
    text = re.sub(r"http\S+|www\S+", "URL", text)

    # Replace user mentions with a common token
    text = re.sub(r"@\w+", "@USER", text)

    # Keep the hashtag word, but remove the # symbol
    text = re.sub(r"#", "", text)

    # Normalize multiple spaces, tabs and new lines
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_data() -> pd.DataFrame:
    """
    Loads the raw Davidson dataset from data/raw/labeled_data.csv.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}\n"
            "Please run src/download_data.py first."
        )

    return pd.read_csv(RAW_DATA_PATH)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the necessary columns, renames them and creates cleaned text.
    """
    required_columns = ["tweet", "class"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    processed_df = df[["tweet", "class"]].copy()

    processed_df = processed_df.rename(
        columns={
            "tweet": "original_tweet",
            "class": "label"
        }
    )

    processed_df["label_name"] = processed_df["label"].map(LABEL_MAP)
    processed_df["text"] = processed_df["original_tweet"].apply(clean_tweet)

    processed_df = processed_df[processed_df["text"].str.len() > 0].copy()

    processed_df = processed_df[
        ["text", "label", "label_name", "original_tweet"]
    ]

    return processed_df


def split_data(df: pd.DataFrame):
    """
    Splits the dataset into train, validation and test sets.

    Final split:
    - train: 70%
    - validation: 15%
    - test: 15%

    Stratification is used because the dataset is imbalanced.
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, validation_df, test_df


def print_class_distribution(name: str, df: pd.DataFrame) -> None:
    """
    Prints class counts and class percentages for a dataset split.
    """
    print(f"\n{name} set")
    print("-" * 40)
    print(f"Shape: {df.shape}")

    print("\nClass counts:")
    print(df["label_name"].value_counts())

    print("\nClass percentages:")
    print((df["label_name"].value_counts(normalize=True) * 100).round(2))


def save_processed_data(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """
    Saves the cleaned full dataset and the three splits.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    full_df.to_csv(PROCESSED_DATA_DIR / "full_cleaned.csv", index=False)
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    validation_df.to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    print("\nProcessed files saved successfully:")
    print(PROCESSED_DATA_DIR / "full_cleaned.csv")
    print(PROCESSED_DATA_DIR / "train.csv")
    print(PROCESSED_DATA_DIR / "validation.csv")
    print(PROCESSED_DATA_DIR / "test.csv")


def main() -> None:
    raw_df = load_data()
    processed_df = preprocess_data(raw_df)

    train_df, validation_df, test_df = split_data(processed_df)

    print_class_distribution("Full", processed_df)
    print_class_distribution("Train", train_df)
    print_class_distribution("Validation", validation_df)
    print_class_distribution("Test", test_df)

    save_processed_data(
        full_df=processed_df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df
    )


if __name__ == "__main__":
    main()