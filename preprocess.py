import os
import pandas as pd
import config


def main():
    if not config.RAW_DATA_FILE.exists():
        print(f"Error: {config.RAW_DATA_FILE} not found. Run dataset_download.py first.")
        return

    df = pd.read_csv(config.RAW_DATA_FILE)
    print(f"Loaded {len(df)} raw passages")

    df = df.dropna(subset=["passage_text"])
    df = df.drop_duplicates(subset=["passage_id"])
    df["passage_text"] = df["passage_text"].str.strip()

    df["word_count"] = df["passage_text"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= config.MIN_WORDS_PER_PASSAGE]
    df = df.reset_index(drop=True)

    df.to_csv(config.PROCESSED_DATA_FILE, index=False)
    print(f"Saved {len(df)} preprocessed passages to {config.PROCESSED_DATA_FILE}")
    print(f"  word_count stats:\n{df['word_count'].describe()}")


if __name__ == "__main__":
    main()
