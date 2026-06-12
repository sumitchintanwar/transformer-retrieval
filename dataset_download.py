import os
import pandas as pd
from datasets import load_dataset


NUM_PASSAGES = 10_000
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "msmarco_passages_raw.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Streaming passages from MS MARCO v1.1 (targeting {NUM_PASSAGES})...")
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True)

    seen = set()
    passages = []
    for sample in dataset:
        for ps in sample["passages"]["passage_text"]:
            if ps not in seen:
                seen.add(ps)
                passages.append(ps)
                if len(passages) % 1000 == 0:
                    print(f"  Collected {len(passages)}/{NUM_PASSAGES} unique passages...")
                if len(passages) >= NUM_PASSAGES:
                    break
        if len(passages) >= NUM_PASSAGES:
            break

    df = pd.DataFrame({
        "passage_id": list(range(len(passages))),
        "passage_text": passages,
    })
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} passages to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
