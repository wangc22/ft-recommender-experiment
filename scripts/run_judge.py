"""Post-hoc LLM-as-Judge for generation runs.

Reads each `outputs/runs/{run}/eval_metrics.json`. For runs with `judge_pending=True`
(generation tasks where the cloud-side run skipped the judge call), invokes the
Anthropic API to score predictions, writes back the eval_metrics.json with real
judge scores, and updates the corresponding row in `outputs/aggregated/all_runs.jsonl`.

Idempotent: re-running won't re-judge runs that already have judge scores
(unless --force is passed).

Run locally:

    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/run_judge.py
    python scripts/analyze_results.py   # composite score now uses real judge
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.evaluation.llm_judge import judge_batch
from src.utils.io import save_json
from src.utils.logging import get_logger

logger = get_logger("run_judge")

ALL_RUNS_PATH = Path("outputs/aggregated/all_runs.jsonl")
RUNS_DIR = Path("outputs/runs")


def _needs_judge(em: dict, force: bool) -> bool:
    if force:
        return True
    return bool(em.get("judge_pending"))


def _patch_one_run(run_dir: Path, force: bool) -> dict | None:
    em_path = run_dir / "eval_metrics.json"
    tm_path = run_dir / "train_metrics.json"
    if not em_path.exists() or not tm_path.exists():
        return None
    em = json.loads(em_path.read_text())
    if "predictions" not in em:
        return None  # not a generation run
    if not _needs_judge(em, force):
        logger.info(f"  skip {run_dir.name} (already judged)")
        return None

    user_queries = [p["user"] for p in em["predictions"]]
    predictions = [p["prediction"] for p in em["predictions"]]

    logger.info(f"  judging {run_dir.name} ({len(predictions)} samples)...")
    try:
        judge_result = judge_batch(user_queries, predictions)
    except Exception as e:
        logger.error(f"  judge_batch failed for {run_dir.name}: {e}")
        return None

    judge_means = judge_result["means"]
    format_pass_rate = em.get("format_pass_rate", 0.0)
    eval_quality = round(
        0.7 * (judge_means["judge_avg"] / 5.0) + 0.3 * format_pass_rate, 4
    )

    em.update({
        "eval_quality": eval_quality,
        "judge_pending": False,
        "judge_avg": judge_means["judge_avg"],
        "judge_helpfulness": judge_means["helpfulness"],
        "judge_format": judge_means["format"],
        "judge_policy_consistency": judge_means["policy_consistency"],
        "n_valid_judge": judge_means["n_valid"],
        "predictions": [
            {**em["predictions"][i], "judge": judge_result["per_sample"][i]["scores"]}
            for i in range(len(predictions))
        ],
    })
    save_json(em, str(em_path))

    tm = json.loads(tm_path.read_text())
    return {
        "task": tm["task"],
        "method": tm["method"],
        "run_type": tm["run_type"],
        "seed": tm["seed"],
        "eval_quality": eval_quality,
        "judge_avg": judge_means["judge_avg"],
    }


def _update_all_runs_jsonl(updates: list[dict]) -> None:
    if not ALL_RUNS_PATH.exists():
        logger.warning(f"{ALL_RUNS_PATH} missing; cannot update aggregate.")
        return
    by_key = {(u["task"], u["method"], u["run_type"], u["seed"]): u for u in updates}
    with ALL_RUNS_PATH.open() as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for r in rows:
        key = (r.get("task"), r.get("method"), r.get("run_type"), r.get("seed"))
        if key in by_key:
            u = by_key[key]
            r["eval_quality_raw"] = u["eval_quality"]
            r["eval_quality_metric_name"] = "judge_avg_plus_format"
            r["judge_pending"] = False
    with ALL_RUNS_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Patched {len(by_key)} row(s) in {ALL_RUNS_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-judge even runs that have judge scores already")
    parser.add_argument("--only-task", default="bitext_support",
                        help="Only judge this task (default: bitext_support)")
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit(
            "ANTHROPIC_API_KEY not set. Export it before running:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    if not RUNS_DIR.exists():
        raise SystemExit(f"{RUNS_DIR} missing. Run scripts/run_matrix.py first.")

    target_run_dirs = [
        d for d in sorted(RUNS_DIR.iterdir())
        if d.is_dir() and (args.only_task is None or d.name.startswith(f"{args.only_task}_"))
    ]
    if not target_run_dirs:
        raise SystemExit(f"No run dirs match task={args.only_task}")

    logger.info(f"Found {len(target_run_dirs)} run dir(s) for task={args.only_task}")
    updates: list[dict] = []
    for d in target_run_dirs:
        u = _patch_one_run(d, args.force)
        if u:
            updates.append(u)

    if updates:
        _update_all_runs_jsonl(updates)
    logger.info(f"\n=== Judge done: {len(updates)} run(s) judged this pass ===")


if __name__ == "__main__":
    main()
