"""Aggregate all_runs.jsonl → composite scores, ranking analysis, plots, summary."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.analysis.composite import compute_composite
from src.analysis.plots import make_all
from src.analysis.ranking import analyze_pilot_vs_larger
from src.utils.io import load_jsonl, save_json, save_jsonl
from src.utils.logging import get_logger

logger = get_logger("analyze")

ALL_RUNS_PATH = "outputs/aggregated/all_runs.jsonl"
OUT_DIR = Path("outputs/aggregated")


def main():
    if not Path(ALL_RUNS_PATH).exists():
        raise SystemExit(f"Missing {ALL_RUNS_PATH}. Run scripts/run_matrix.py first.")
    raw = load_jsonl(ALL_RUNS_PATH)
    logger.info(f"Loaded {len(raw)} raw run records.")

    # Deduplicate on (task, method, run_type, seed) — keeping the LAST entry per
    # key. Necessary because run_one appends to all_runs.jsonl and reruns for the
    # same triple would otherwise inflate composite scores.
    by_key: dict = {}
    for r in raw:
        key = (r.get("task"), r.get("method"), r.get("run_type"), r.get("seed"))
        by_key[key] = r
    records = list(by_key.values())
    n_dropped = len(raw) - len(records)
    if n_dropped > 0:
        logger.warning(
            f"Dropped {n_dropped} duplicate row(s); kept {len(records)} unique runs. "
            "Last-write-wins per (task, method, run_type, seed)."
        )

    scored = compute_composite(records)

    # Mark recommender selections (pilot top-1 by composite per task).
    by_task_pilot = {}
    for r in scored:
        if r["run_type"] == "pilot":
            by_task_pilot.setdefault(r["task"], []).append(r)
    for task, recs in by_task_pilot.items():
        winner = max(recs, key=lambda r: r["composite_score"])
        for r in scored:
            if (r["task"] == task and r["run_type"] == "pilot"
                    and r["method"] == winner["method"]):
                r["selected_by_recommender"] = True

    save_jsonl(scored, str(OUT_DIR / "all_runs_scored.jsonl"))
    ranking = analyze_pilot_vs_larger(scored)
    save_json(ranking, str(OUT_DIR / "pilot_vs_larger.json"))
    make_all(scored, ranking)

    logger.info("\n=== Summary ===")
    for task, info in ranking.items():
        if "error" in info:
            logger.info(f"{task}: ERROR {info['error']}")
            continue
        logger.info(
            f"{task}: pilot_top1={info['pilot_top1']} "
            f"larger_top1={info['larger_top1']} "
            f"top1_match={info['top1_match']} "
            f"top2_overlap={info['top2_overlap']} "
            f"regret_rel={info['regret_relative']:.2%} "
            f"success={info['success']}"
        )

    summary_md = ["# Recommender Validation Summary\n"]
    for task, info in ranking.items():
        if "error" in info:
            summary_md.append(f"## {task}\n\nERROR: {info['error']}\n")
            continue
        summary_md.append(
            f"## {task}\n"
            f"- Pilot top-1 (composite): {info['pilot_top1']}, top-2={info['pilot_top2']}\n"
            f"- Larger top-1 (eval quality): {info['larger_top1']}\n"
            f"- Top-1 match: {info['top1_match']}\n"
            f"- Top-2 overlap: {info['top2_overlap']}\n"
            f"- Regret (relative): {info['regret_relative']:.2%}\n"
            f"- Spearman: {info['spearman']} | Kendall tau: {info['kendall_tau']}\n"
            f"- **Decision: {'PASS' if info['success'] else 'FAIL'}**\n"
        )
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports/summary.md").write_text("\n".join(summary_md), encoding="utf-8")
    logger.info("Wrote outputs/reports/summary.md and outputs/figures/*.png")


if __name__ == "__main__":
    main()
