from logger import get_logger

logger = get_logger(__name__)
import os

import pandas as pd
from datasets import load_dataset

import config


def main():
    logger.info(
        f"Streaming passages from MS MARCO v1.1 (targeting {config.NUM_PASSAGES_TO_DOWNLOAD})..."
    )
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True)

    seen = set()
    passages = []
    for sample in dataset:
        for ps in sample["passages"]["passage_text"]:
            if ps not in seen:
                seen.add(ps)
                passages.append(ps)
                if len(passages) % 1000 == 0:
                    logger.info(
                        f"  Collected {len(passages)}/{config.NUM_PASSAGES_TO_DOWNLOAD} unique passages..."
                    )
                if len(passages) >= config.NUM_PASSAGES_TO_DOWNLOAD:
                    break
        if len(passages) >= config.NUM_PASSAGES_TO_DOWNLOAD:
            break

    df = pd.DataFrame(
        {
            "passage_id": list(range(len(passages))),
            "passage_text": passages,
        }
    )
    df.to_csv(config.RAW_DATA_FILE, index=False)
    logger.info(f"Saved {len(df)} passages to {config.RAW_DATA_FILE}")


if __name__ == "__main__":
    main()
