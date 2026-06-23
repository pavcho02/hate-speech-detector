from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RAW_DATA_PATH = Path("data/raw/labeled_data.csv")
FIGURES_DIR = Path("reports/figures")

LABEL_MAP = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither"
}


def load_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_PATH}. "
            "Run src/01_download_data.py first."
        )

    return pd.read_csv(RAW_DATA_PATH)


def explore_data(df):
    print("Dataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate tweets:")
    print(df["tweet"].duplicated().sum())

    df["label_name"] = df["class"].map(LABEL_MAP)

    print("\nClass distribution:")
    class_distribution = df["label_name"].value_counts()
    print(class_distribution)

    print("\nClass distribution percentage:")
    print((df["label_name"].value_counts(normalize=True) * 100).round(2))

    df["tweet_length"] = df["tweet"].astype(str).apply(len)
    df["word_count"] = df["tweet"].astype(str).apply(lambda text: len(text.split()))

    print("\nTweet length statistics:")
    print(df[["tweet_length", "word_count"]].describe())

    return df


def save_class_distribution_plot(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    class_counts = df["label_name"].value_counts()

    class_counts.plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_path = FIGURES_DIR / "class_distribution.png"
    plt.savefig(output_path)
    plt.close()

    print(f"\nClass distribution plot saved to: {output_path}")


def save_word_count_plot(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df["word_count"].plot(kind="hist", bins=40)
    plt.title("Tweet Word Count Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()

    output_path = FIGURES_DIR / "word_count_distribution.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Word count plot saved to: {output_path}")


if __name__ == "__main__":
    data = load_data()
    data = explore_data(data)

    save_class_distribution_plot(data)
    save_word_count_plot(data)