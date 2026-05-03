"""Run the full PEFT-method × task × scale matrix as subprocess invocations of run_one.

Subprocess isolation prevents GPU memory accumulation across runs.
"""
import argparse
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.utils.config import load_yaml
from src.utils.logging import get_logger

logger = get_logger("run_matrix")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/matrix.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-task", default=None,
                        help="Run only one task (banking77|bitext_support|cuad)")
    parser.add_argument("--only-method", default=None,
                        help="Run only one method (lora|dora|qlora|lora_plus)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    tasks = [args.only_task] if args.only_task else cfg["tasks"]
    methods = [args.only_method] if args.only_method else cfg["methods"]
    scales = cfg["scales"]
    seed = cfg["seed"]
    sleep_s = cfg.get("inter_run_sleep_seconds", 5)
    fail_fast = cfg.get("fail_fast", False)
    model_path = cfg.get("model_path", "models/Qwen2.5-1.5B-Instruct")

    runs = list(product(tasks, scales, methods))
    logger.info(f"Total runs: {len(runs)}")

    if args.dry_run:
        for i, (t, s, m) in enumerate(runs, 1):
            print(f"  [{i}/{len(runs)}] task={t} scale={s} method={m} seed={seed}")
        return

    # Pre-flight: warn if all_runs.jsonl already has rows (would cause duplicates
    # in analyze_results.py and corrupt composite scores).
    all_runs_path = Path("outputs/aggregated/all_runs.jsonl")
    if all_runs_path.exists() and all_runs_path.stat().st_size > 0:
        n_existing = sum(1 for _ in open(all_runs_path))
        logger.warning(
            f"{all_runs_path} already contains {n_existing} rows. New rows will be "
            "APPENDED — analyze_results.py will see duplicates if you re-run the same "
            "(task, method, scale, seed) triples. Delete the file or use --only-* "
            "filters to avoid overlap."
        )

    overall_t0 = time.time()
    completed, failed = 0, 0
    for i, (task, scale, method) in enumerate(runs, 1):
        logger.info(f"\n{'='*60}\n[{i}/{len(runs)}] task={task} scale={scale} method={method}\n{'='*60}")
        cmd = [
            sys.executable, "-m", "src.training.run_one",
            "--task", task, "--method", method,
            "--scale", scale, "--seed", str(seed),
            "--model-path", model_path,
        ]
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        dt = round(time.time() - t0, 1)
        if rc != 0:
            failed += 1
            logger.error(f"Run failed (rc={rc}) in {dt}s")
            if fail_fast:
                sys.exit(1)
        else:
            completed += 1
            logger.info(f"Run done in {dt}s")
        if i < len(runs):
            time.sleep(sleep_s)

    total = round(time.time() - overall_t0, 1)
    logger.info(f"\n=== Matrix done: {completed} completed / {failed} failed / {total}s total ===")

    # Per-run final summary table read from all_runs.jsonl, so the user has a
    # one-shot scannable view at the bottom of the cloud log.
    if all_runs_path.exists() and all_runs_path.stat().st_size > 0:
        import json as _json
        records = [_json.loads(l) for l in open(all_runs_path) if l.strip()]
        # Keep only the runs from THIS matrix invocation (filter by tasks/methods/scales)
        in_scope = [r for r in records
                    if r.get("task") in tasks and r.get("method") in methods
                    and r.get("run_type") in scales and r.get("seed") == seed]
        # Last-write-wins per (task, method, run_type, seed)
        by_key: dict = {}
        for r in in_scope:
            by_key[(r["task"], r["method"], r["run_type"], r["seed"])] = r
        in_scope = list(by_key.values())

        if in_scope:
            print("\n" + "=" * 88)
            print("=== ALL RUNS SUMMARY TABLE ===")
            print("-" * 88)
            print(f"  {'task':<16} {'scale':<7} {'method':<10} "
                  f"{'eval_q':>7} {'train_l':>8} {'eval_l':>8} "
                  f"{'time_s':>7} {'mem_MB':>8}")
            print("-" * 88)
            in_scope.sort(key=lambda r: (r["task"], r["run_type"], r["method"]))
            for r in in_scope:
                print(f"  {r['task']:<16} {r['run_type']:<7} {r['method']:<10} "
                      f"{r['eval_quality_raw']:>7.4f} "
                      f"{r['train_loss']:>8.4f} {r['eval_loss']:>8.4f} "
                      f"{r['training_time']:>7.1f} {r['memory_cost']:>8.0f}")
            print("=" * 88)
            print(f"  Next: python scripts/analyze_results.py")
            print("=" * 88)


if __name__ == "__main__":
    main()
