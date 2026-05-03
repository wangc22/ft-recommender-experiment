"""CUAD → pilot/larger/val/test jsonl with contract-level split.

We use cuad-qa (SQuAD-style). Each (question, context, answer) tuple has a
clause_type derived from the question template. We keep only positive spans
(non-empty answer) and treat (clause_text, clause_type) as the classification
record. Splits are made at the contract level to prevent leakage.
"""
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from src.utils.io import save_jsonl, save_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

DATASET_ID = "theatticusproject/cuad-qa"
OUT_DIR = Path("data/processed/cuad")

# 10 categories that exist in CUAD and cover commercial/legal contract review.
TARGET_LABELS = [
    "Termination For Convenience",
    "Governing Law",
    "Non-Compete",
    "Exclusivity",
    "Cap On Liability",
    "Anti-Assignment",
    "Change Of Control",
    "Warranty Duration",
    "License Grant",
    "Audit Rights",
]

PILOT_TRAIN_N = 200
LARGER_TRAIN_N = 800
VAL_N = 200
TEST_N = 200
SEED = 42

CONTRACT_TRAIN_RATIO = 0.70
CONTRACT_VAL_RATIO = 0.15

QUESTION_LABEL_RE = re.compile(r"related to ['\"‘’]([^'\"‘’]+?)['\"‘’]")


def _extract_label_from_question(question: str) -> str | None:
    m = QUESTION_LABEL_RE.search(question)
    if not m:
        return None
    return m.group(1).strip()


def prepare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, cache_dir="data/raw", trust_remote_code=True)

    all_recs = []
    for split_name in ds.keys():
        for r in ds[split_name]:
            label = _extract_label_from_question(r["question"])
            if label is None or label not in TARGET_LABELS:
                continue
            answers = r.get("answers", {})
            spans = answers.get("text", []) if isinstance(answers, dict) else []
            if not spans:
                continue
            text = spans[0].strip()
            if len(text) < 20:
                continue
            all_recs.append({
                "input": text[:2000],
                "label": label,
                "label_idx": TARGET_LABELS.index(label),
                "contract_id": r["title"],
            })

    if not all_recs:
        raise RuntimeError("CUAD: no records collected. Check question regex / dataset schema.")

    label_dist = Counter(r["label"] for r in all_recs)
    logger.info(f"CUAD raw label distribution: {dict(label_dist)}")

    # Drop labels with too few examples (< 30) and possibly fall back to 8 cats.
    rare = [lab for lab, n in label_dist.items() if n < 30]
    if rare:
        logger.warning(f"Dropping rare labels (< 30 examples): {rare}")
        all_recs = [r for r in all_recs if r["label"] not in rare]

    final_labels = sorted({r["label"] for r in all_recs})
    label_to_idx = {lab: i for i, lab in enumerate(final_labels)}
    for r in all_recs:
        r["label_idx"] = label_to_idx[r["label"]]
    save_json(final_labels, str(OUT_DIR / "labels.json"))
    logger.info(f"Final label set ({len(final_labels)}): {final_labels}")

    # Contract-level split.
    contracts = sorted({r["contract_id"] for r in all_recs})
    rng = random.Random(SEED)
    rng.shuffle(contracts)
    n = len(contracts)
    n_train = int(n * CONTRACT_TRAIN_RATIO)
    n_val = int(n * CONTRACT_VAL_RATIO)
    train_contracts = set(contracts[:n_train])
    val_contracts = set(contracts[n_train:n_train + n_val])
    test_contracts = set(contracts[n_train + n_val:])

    # Sanity: no overlap.
    assert not (train_contracts & val_contracts)
    assert not (train_contracts & test_contracts)
    assert not (val_contracts & test_contracts)

    save_json({
        "train_contracts": sorted(train_contracts),
        "val_contracts": sorted(val_contracts),
        "test_contracts": sorted(test_contracts),
        "n_contracts_total": n,
    }, str(OUT_DIR / "split_manifest.json"))

    by_split = defaultdict(list)
    for r in all_recs:
        cid = r["contract_id"]
        if cid in train_contracts:
            by_split["train"].append(r)
        elif cid in val_contracts:
            by_split["val"].append(r)
        elif cid in test_contracts:
            by_split["test"].append(r)

    for k, recs in by_split.items():
        rng.shuffle(recs)
        logger.info(f"CUAD {k}: {len(recs)} records, "
                    f"label dist: {dict(Counter(r['label'] for r in recs))}")

    if len(by_split["train"]) < LARGER_TRAIN_N:
        raise RuntimeError(
            f"CUAD train pool too small: {len(by_split['train'])} < {LARGER_TRAIN_N}")

    larger_train = _balanced_sample(by_split["train"], LARGER_TRAIN_N, final_labels, rng)
    pilot_train = larger_train[:PILOT_TRAIN_N]
    val = by_split["val"][:VAL_N]
    test = by_split["test"][:TEST_N]

    if len(val) < VAL_N or len(test) < TEST_N:
        logger.warning(f"CUAD val/test undersized: val={len(val)} test={len(test)}")

    save_jsonl(pilot_train, str(OUT_DIR / "pilot_train.jsonl"))
    save_jsonl(larger_train, str(OUT_DIR / "larger_train.jsonl"))
    save_jsonl(val, str(OUT_DIR / "val.jsonl"))
    save_jsonl(test, str(OUT_DIR / "test.jsonl"))

    assert pilot_train == larger_train[:PILOT_TRAIN_N], "pilot ⊄ larger"
    logger.info(f"CUAD: pilot={len(pilot_train)} larger={len(larger_train)} "
                f"val={len(val)} test={len(test)} | labels={len(final_labels)}")


def _balanced_sample(records, target_n, labels, rng) -> list:
    """Round-robin across labels until we hit target_n."""
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)
    for lab in by_label:
        rng.shuffle(by_label[lab])

    out = []
    cursors = {lab: 0 for lab in labels}
    while len(out) < target_n:
        progressed = False
        for lab in labels:
            if cursors[lab] < len(by_label.get(lab, [])):
                out.append(by_label[lab][cursors[lab]])
                cursors[lab] += 1
                progressed = True
                if len(out) >= target_n:
                    break
        if not progressed:
            break
    return out


if __name__ == "__main__":
    prepare()
