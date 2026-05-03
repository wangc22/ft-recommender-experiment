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


if __name__ == "__main__":
    main()
