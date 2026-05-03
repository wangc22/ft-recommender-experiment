"""Run the full PEFT-method × task × scale matrix as subprocess invocations of run_one.

Subprocess isolation prevents GPU memory accumulation across runs.

**Resumability**: this script reads outputs/aggregated/all_runs.jsonl on startup
and SKIPS any (task, method, scale, seed) triple that already has a recorded
result. Use --force to re-run everything, --no-skip-completed to disable the
cache check, or delete all_runs.jsonl to start fresh.
"""
import argparse
import json
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


def _load_completed_keys(jsonl_path: Path) -> set[tuple]:
    """Return set of (task, method, run_type, seed) tuples already recorded."""
    keys: set[tuple] = set()
    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        return keys
    for line in jsonl_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            keys.add((r.get("task"), r.get("method"), r.get("run_type"), r.get("seed")))
        except json.JSONDecodeError:
            logger.warning(f"Skipping unparseable line in {jsonl_path}")
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/matrix.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-task", default=None,
                        help="Run only one task (banking77|bitext_support|cuad)")
    parser.add_argument("--only-method", default=None,
                        help="Run only one method (lora|dora|qlora|lora_plus)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all triples even if they already have a recorded result.")
    parser.add_argument("--no-skip-completed", action="store_true",
                        help="Don't read all_runs.jsonl to skip completed runs (alias for --force).")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    tasks = [args.only_task] if args.only_task else cfg["tasks"]
    methods = [args.only_method] if args.only_method else cfg["methods"]
    scales = cfg["scales"]
    seed = cfg["seed"]
    sleep_s = cfg.get("inter_run_sleep_seconds", 5)
    fail_fast = cfg.get("fail_fast", False)
    model_path = cfg.get("model_path", "models/Qwen2.5-1.5B-Instruct")

    all_runs_path = Path("outputs/aggregated/all_runs.jsonl")
    skip_completed = not (args.force or args.no_skip_completed)
    completed_keys = _load_completed_keys(all_runs_path) if skip_completed else set()

    full_runs = list(product(tasks, scales, methods))
    if skip_completed and completed_keys:
        runs_to_do = [
            (t, s, m) for (t, s, m) in full_runs
            if (t, m, s, seed) not in completed_keys
        ]
    else:
        runs_to_do = full_runs

    n_total = len(full_runs)
    n_skip = n_total - len(runs_to_do)
    logger.info(f"Matrix scope: {n_total} runs total | {n_skip} already done (skipping) | {len(runs_to_do)} to run")

    if args.dry_run:
        if n_skip:
            print(f"\n[DRY-RUN] Skipping {n_skip} already-completed runs (in all_runs.jsonl):")
            for (t, s, m) in full_runs:
                if (t, m, s, seed) in completed_keys:
                    print(f"  · SKIP   task={t} scale={s} method={m}")
            print("")
        print(f"[DRY-RUN] Will execute {len(runs_to_do)} runs:")
        for i, (t, s, m) in enumerate(runs_to_do, 1):
            print(f"  [{i}/{len(runs_to_do)}] task={t} scale={s} method={m} seed={seed}")
        return

    if not runs_to_do:
        logger.info("Nothing to do — all configured runs already complete in all_runs.jsonl.")
        logger.info(f"To force a re-run: python scripts/run_matrix.py --force")
        return

    overall_t0 = time.time()
    completed, failed = 0, 0
    for i, (task, scale, method) in enumerate(runs_to_do, 1):
        logger.info(
            f"\n{'='*60}\n[{i}/{len(runs_to_do)}] task={task} scale={scale} "
            f"method={method}{'  (resuming, skipped ' + str(n_skip) + ' completed)' if i == 1 and n_skip else ''}\n{'='*60}"
        )
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
        if i < len(runs_to_do):
            time.sleep(sleep_s)

    total = round(time.time() - overall_t0, 1)
    logger.info(
        f"\n=== Matrix done: {completed} new completed / {failed} failed / "
        f"{n_skip} previously-completed-skipped / {total}s wall-clock ==="
    )

    # Per-run final summary table read from all_runs.jsonl, so the user has a
    # one-shot scannable view at the bottom of the cloud log.
    if all_runs_path.exists() and all_runs_path.stat().st_size > 0:
        records = [json.loads(l) for l in open(all_runs_path) if l.strip()]
        # Keep only the runs in current scope (this matrix's tasks/methods/scales)
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
