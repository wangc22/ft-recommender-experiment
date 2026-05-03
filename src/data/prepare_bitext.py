"""Bitext customer support → pilot/larger/val/test jsonl with stable seed=42."""
import random
from pathlib import Path

from datasets import load_dataset

from src.utils.io import save_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
OUT_DIR = Path("data/processed/bitext_support")

PILOT_TRAIN_N = 100
LARGER_TRAIN_N = 500
VAL_N = 100
TEST_N = 100
SEED = 42


def prepare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, cache_dir="data/raw", split="train",
                      trust_remote_code=True)

    records = [
        {"messages": [
            {"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": r["response"]},
        ]}
        for r in ds
    ]

    rng = random.Random(SEED)
    rng.shuffle(records)

    needed = LARGER_TRAIN_N + VAL_N + TEST_N
    if len(records) < needed:
        raise RuntimeError(f"Bitext too small: {len(records)} < {needed}")

    larger_train = records[:LARGER_TRAIN_N]
    pilot_train = larger_train[:PILOT_TRAIN_N]
    val = records[LARGER_TRAIN_N:LARGER_TRAIN_N + VAL_N]
    test = records[LARGER_TRAIN_N + VAL_N:LARGER_TRAIN_N + VAL_N + TEST_N]

    save_jsonl(pilot_train, str(OUT_DIR / "pilot_train.jsonl"))
    save_jsonl(larger_train, str(OUT_DIR / "larger_train.jsonl"))
    save_jsonl(val, str(OUT_DIR / "val.jsonl"))
    save_jsonl(test, str(OUT_DIR / "test.jsonl"))

    assert pilot_train == larger_train[:PILOT_TRAIN_N], "pilot ⊄ larger"
    logger.info(f"Bitext: pilot={len(pilot_train)} larger={len(larger_train)} "
                f"val={len(val)} test={len(test)}")


if __name__ == "__main__":
    prepare()
