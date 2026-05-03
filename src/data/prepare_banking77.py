"""BANKING77 → pilot/larger/val/test jsonl with stable seed=42 split."""
import random
from pathlib import Path

from datasets import load_dataset

from src.utils.io import save_jsonl, save_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

DATASET_ID = "PolyAI/banking77"
OUT_DIR = Path("data/processed/banking77")

PILOT_TRAIN_N = 200
LARGER_TRAIN_N = 1000
VAL_N = 200
TEST_N = 200
SEED = 42


def prepare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, cache_dir="data/raw", trust_remote_code=True)

    label_names = ds["train"].features["label"].names
    save_json(label_names, str(OUT_DIR / "labels.json"))
    logger.info(f"Saved {len(label_names)} labels.")

    train_recs = [{"input": r["text"],
                   "label": label_names[r["label"]],
                   "label_idx": r["label"]} for r in ds["train"]]
    test_recs = [{"input": r["text"],
                  "label": label_names[r["label"]],
                  "label_idx": r["label"]} for r in ds["test"]]

    rng = random.Random(SEED)
    rng.shuffle(train_recs)
    rng.shuffle(test_recs)

    if len(train_recs) < LARGER_TRAIN_N + VAL_N:
        raise RuntimeError(f"Train set too small: {len(train_recs)}")

    larger_train = train_recs[:LARGER_TRAIN_N]
    pilot_train = larger_train[:PILOT_TRAIN_N]
    val = train_recs[LARGER_TRAIN_N:LARGER_TRAIN_N + VAL_N]
    test = test_recs[:TEST_N]

    save_jsonl(pilot_train, str(OUT_DIR / "pilot_train.jsonl"))
    save_jsonl(larger_train, str(OUT_DIR / "larger_train.jsonl"))
    save_jsonl(val, str(OUT_DIR / "val.jsonl"))
    save_jsonl(test, str(OUT_DIR / "test.jsonl"))

    assert pilot_train == larger_train[:PILOT_TRAIN_N], "pilot ⊄ larger"
    logger.info(f"BANKING77: pilot={len(pilot_train)} larger={len(larger_train)} "
                f"val={len(val)} test={len(test)} | labels={len(label_names)}")


if __name__ == "__main__":
    prepare()
