import os
import pandas as pd


INPUT_FILE = os.path.join("data", "msmarco_passages_raw.csv")
OUTPUT_FILE = os.path.join("data", "msmarco_passages.csv")
MIN_WORDS = 5


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run dataset_download.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} raw passages")

    df = df.dropna(subset=["passage_text"])
    df = df.drop_duplicates(subset=["passage_id"])
    df["passage_text"] = df["passage_text"].str.strip()

    df["word_count"] = df["passage_text"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= MIN_WORDS]
    df = df.reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} preprocessed passages to {OUTPUT_FILE}")
    print(f"  word_count stats:\n{df['word_count'].describe()}")


if __name__ == "__main__":
    main()
