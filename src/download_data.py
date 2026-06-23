from pathlib import Path
import pandas as pd


DATA_URL = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_PATH = RAW_DATA_DIR / "labeled_data.csv"


def download_dataset():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset...")
    df = pd.read_csv(DATA_URL)

    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Dataset saved to: {RAW_DATA_PATH}")
    print(f"Dataset shape: {df.shape}")


if __name__ == "__main__":
    download_dataset()