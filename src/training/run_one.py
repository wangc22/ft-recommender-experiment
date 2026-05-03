"""Single fine-tune + eval run: (task, method, scale, seed) → metrics jsonl record."""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model, prepare_model_for_kbit_training

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.data.formatters import (
    format_classification_for_sft,
    format_generation_for_sft,
)
from src.evaluation.classification import evaluate_classification
from src.evaluation.generation import evaluate_generation
from src.evaluation.metrics import get_peak_memory_mb, reset_peak_memory
from src.training.peft_factory import (
    SUPPORTED_METHODS,
    build_peft,
    count_trainable_params,
)
from src.utils.config import load_yaml
from src.utils.io import append_jsonl, ensure_dir, load_json, load_jsonl, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("run_one")

DEFAULT_MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"
ALL_RUNS_PATH = "outputs/aggregated/all_runs.jsonl"


def _data_paths(task: str, scale: str) -> dict:
    base = Path("data/processed") / task
    train_file = "pilot_train.jsonl" if scale == "pilot" else "larger_train.jsonl"
    return {
        "train": str(base / train_file),
        "val": str(base / "val.jsonl"),
        "labels": str(base / "labels.json"),
    }


def _format_for_sft(records, task_kind, labels):
    if task_kind == "classification":
        return [format_classification_for_sft(r, labels) for r in records]
    if task_kind == "generation":
        return [format_generation_for_sft(r) for r in records]
    raise ValueError(f"Unknown task_kind: {task_kind}")


def _device_info():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def run_one(task: str, method: str, scale: str, seed: int = 42,
            model_path: str = DEFAULT_MODEL_PATH) -> dict:
    if method.lower() not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}, got {method!r}")

    device_kind, compute_dtype = _device_info()
    if method.lower() == "qlora" and device_kind != "cuda":
        raise RuntimeError(
            f"QLoRA requires CUDA + bitsandbytes. Detected device={device_kind}. "
            "Run QLoRA on a GPU instance."
        )

    set_seed(seed)
    task_cfg = load_yaml(f"configs/tasks/{task}.yaml")
    paths = _data_paths(task, scale)
    task_kind = task_cfg["kind"]
    labels = load_json(paths["labels"]) if task_kind == "classification" else None

    train_recs = load_jsonl(paths["train"])
    val_recs = load_jsonl(paths["val"])
    logger.info(f"[{task}/{scale}/{method}] train={len(train_recs)} val={len(val_recs)} "
                f"device={device_kind}")

    run_name = f"{task}_{scale}_{method}_seed{seed}"
    run_dir = Path("outputs/runs") / run_name
    ensure_dir(str(run_dir))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_fmt = _format_for_sft(train_recs, task_kind, labels)
    val_fmt = _format_for_sft(val_recs, task_kind, labels)
    train_ds = Dataset.from_list(train_fmt)
    val_ds = Dataset.from_list(val_fmt)

    # Build PEFT plan; this tells us if we need quantized model loading
    # and/or a custom optimizer factory.
    from transformers import AutoConfig
    plan = build_peft(method, AutoConfig.from_pretrained(model_path),
                      compute_dtype=compute_dtype)
    logger.info(f"PEFT plan: {plan.method} | {plan.notes or '-'}")

    model_load_kwargs = {"torch_dtype": compute_dtype, **plan.model_kwargs}
    if device_kind == "cuda":
        model_load_kwargs["device_map"] = "auto"
    logger.info(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_load_kwargs)
    if device_kind == "mps" and "device_map" not in model_load_kwargs:
        model = model.to("mps")

    if method.lower() == "qlora":
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, plan.lora_config)
    param_stats = count_trainable_params(model)
    logger.info(f"Trainable params: {param_stats}")

    tr = task_cfg["training"]
    base_lr = float(tr.get("learning_rate", 2e-4))
    sft_args = SFTConfig(
        output_dir=str(run_dir / "checkpoint"),
        num_train_epochs=tr.get(f"{scale}_epochs", tr.get("epochs", 1)),
        per_device_train_batch_size=tr.get("batch_size", 4),
        per_device_eval_batch_size=tr.get("batch_size", 4),
        gradient_accumulation_steps=tr.get("gradient_accumulation_steps", 4),
        learning_rate=base_lr,
        warmup_ratio=tr.get("warmup_ratio", 0.05),
        weight_decay=tr.get("weight_decay", 0.01),
        max_length=task_cfg.get("max_length", 512),
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_steps=20,
        report_to="none",
        bf16=(device_kind == "cuda"),
        seed=seed,
    )

    optimizers = (None, None)
    if plan.optimizer_factory is not None:
        opt = plan.optimizer_factory(model, base_lr)
        optimizers = (opt, None)
        logger.info(f"Custom optimizer injected for {method}")

    reset_peak_memory()
    trainer = SFTTrainer(
        model=model, args=sft_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        optimizers=optimizers,
    )
    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0
    eval_loss_metrics = trainer.evaluate()
    peak_mem = get_peak_memory_mb()

    train_loss = float(train_result.training_loss)
    eval_loss = float(eval_loss_metrics.get("eval_loss", float("nan")))
    overfit_gap = eval_loss - train_loss

    train_metrics = {
        "task": task, "method": method, "run_type": scale, "seed": seed,
        "train_size": len(train_recs), "val_size": len(val_recs),
        "training_time": round(train_time, 1),
        "memory_cost": round(peak_mem, 1),
        "train_loss": round(train_loss, 4),
        "eval_loss": round(eval_loss, 4),
        "overfit_gap": round(overfit_gap, 4),
        "trainable_params": param_stats["trainable_params"],
        "total_params": param_stats["total_params"],
    }
    save_json(train_metrics, str(run_dir / "train_metrics.json"))

    logger.info(f"[{task}/{scale}/{method}] eval on val ({len(val_recs)} samples)...")
    if task_kind == "classification":
        eval_metrics = evaluate_classification(trainer.model, tokenizer, val_recs, labels)
        eval_quality_raw = eval_metrics["macro_f1"]
        eval_quality_metric = "macro_f1"
    else:
        eval_metrics = evaluate_generation(trainer.model, tokenizer, val_recs)
        eval_quality_raw = eval_metrics["eval_quality"]
        eval_quality_metric = "judge_avg_plus_format"
    save_json(eval_metrics, str(run_dir / "eval_metrics.json"))

    judge_pending = bool(eval_metrics.get("judge_pending", False))
    record = {
        "task": task,
        "dataset": task_cfg["dataset_id"],
        "method": method,
        "config": method,
        "run_type": scale,
        "seed": seed,
        "train_size": len(train_recs),
        "val_size": len(val_recs),
        "test_size": 0,
        "eval_quality_raw": eval_quality_raw,
        "eval_quality_metric_name": eval_quality_metric,
        "training_time": round(train_time, 1),
        "memory_cost": round(peak_mem, 1),
        "train_loss": round(train_loss, 4),
        "eval_loss": round(eval_loss, 4),
        "overfit_gap": round(overfit_gap, 4),
        "trainable_params": param_stats["trainable_params"],
        "composite_score": None,
        "selected_by_recommender": False,
        "judge_pending": judge_pending,
    }
    append_jsonl(record, ALL_RUNS_PATH)

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    _print_run_summary(record, eval_metrics, task_kind)
    return record


def _print_run_summary(record: dict, eval_metrics: dict, task_kind: str) -> None:
    """Pretty-print a per-run summary so 24-run logs are scannable in the cloud.

    Format: a fenced block with key metrics + heuristic sanity warnings (✓ ok,
    ⚠ suspicious, ✗ likely broken). The block is also picked up easily by
    `grep '=== RUN SUMMARY ==='` for post-hoc log analysis.
    """
    rk = f"[{record['task']}/{record['run_type']}/{record['method']}]"
    quality_warn = _sanity_quality(record, task_kind)
    loss_warn = _sanity_loss(record["train_loss"], record["eval_loss"])
    mem_warn = _sanity_memory(record["memory_cost"])
    overfit_warn = _sanity_overfit(record["overfit_gap"])

    lines = [
        "",
        "=" * 72,
        f"=== RUN SUMMARY {rk} ===",
        "-" * 72,
        f"  eval_quality_raw   : {record['eval_quality_raw']:.4f}  "
        f"({record['eval_quality_metric_name']}) {quality_warn}",
        f"  train_loss         : {record['train_loss']:.4f} {loss_warn[0]}",
        f"  eval_loss          : {record['eval_loss']:.4f} {loss_warn[1]}",
        f"  overfit_gap        : {record['overfit_gap']:+.4f} {overfit_warn}",
        f"  training_time      : {record['training_time']:.1f}s",
        f"  memory_cost        : {record['memory_cost']:.0f} MB {mem_warn}",
        f"  trainable_params   : {record['trainable_params']:,}  "
        f"({record['train_size']} train / {record['val_size']} val)",
    ]
    if task_kind == "classification":
        lines.append(
            f"  invalid_predictions: {eval_metrics.get('n_invalid', '?')} / "
            f"{eval_metrics.get('n_total', '?')}  "
            f"(accuracy={eval_metrics.get('accuracy', '?'):.4f})"
        )
    else:
        lines.append(
            f"  judge / format     : avg={eval_metrics.get('judge_avg', '?')}  "
            f"format_pass={eval_metrics.get('format_pass_rate', '?')}  "
            f"n_valid_judge={eval_metrics.get('n_valid_judge', '?')}/"
            f"{eval_metrics.get('n_total', '?')}"
        )
    lines.append("=" * 72)
    lines.append("")
    print("\n".join(lines))


def _sanity_quality(record: dict, task_kind: str) -> str:
    """Heuristic ranges based on Qwen2.5-1.5B + LoRA-family baselines on these tasks.

    Pilot bands are intentionally wide-low: 200 samples / 1 epoch on a 77-class
    or 10-class problem is barely above random baseline; the experiment cares
    about *relative* ranking between PEFT methods, not absolute quality. We
    only flag ✗ when the score is so low it suggests a code bug (e.g. label
    parser broken, prompt truncated, etc).
    """
    q = record["eval_quality_raw"]
    scale = record["run_type"]
    task = record["task"]
    # Random baselines: banking77 macro-F1 ≈ 1/77 = 0.013; cuad 1/10 = 0.1;
    # bitext_support is generation, no random baseline (proxy quality 0-1).
    bands = {
        "banking77":      {"pilot": (0.05, 0.65),  "larger": (0.30, 0.80)},
        "cuad":           {"pilot": (0.15, 0.75),  "larger": (0.40, 0.85)},
        "bitext_support": {"pilot": (0.30, 0.90),  "larger": (0.40, 0.95)},
    }
    lo, hi = bands.get(task, {}).get(scale, (0.0, 1.0))
    # ✗ red flag: below 1/3 of lower band → likely code bug
    if q < lo / 3:
        return f"✗ suspicious (expected ≥ {lo:.2f}; check predict_batch / data)"
    if q < lo:
        return f"⚠ below expected band [{lo:.2f}, {hi:.2f}] (could be normal for pilot)"
    if q > hi:
        return f"⚠ above expected band [{lo:.2f}, {hi:.2f}] (sanity check?)"
    return "✓"


def _sanity_loss(train_loss: float, eval_loss: float) -> tuple[str, str]:
    train_tag = "✓"
    eval_tag = "✓"
    if train_loss > 5.0:
        train_tag = "✗ unusually high (>5.0; possibly diverged)"
    elif train_loss > 3.5:
        train_tag = "⚠ on the high side"
    if eval_loss > 5.0:
        eval_tag = "✗ unusually high (>5.0)"
    elif eval_loss > 3.5:
        eval_tag = "⚠ on the high side"
    if eval_loss != eval_loss:  # NaN
        eval_tag = "✗ NaN — training likely diverged"
    return train_tag, eval_tag


def _sanity_memory(mb: float) -> str:
    if mb < 1000:
        return "⚠ unusually low (CPU/MPS measurement may be unreliable)"
    if mb > 20000:
        return "✗ over 20GB — investigate OOM risk"
    if mb > 12000:
        return "⚠ above 12GB"
    return "✓"


def _sanity_overfit(gap: float) -> str:
    """Note: in pilot scale (1 epoch, ~50 steps), `train_loss` is averaged
    across early high-loss steps and `eval_loss` is computed AFTER training,
    so a negative gap of up to ~-1.5 is normal — not data leakage. Only
    extreme negative gaps (< -2.0) suggest something genuinely off.
    """
    if gap < -2.0:
        return "⚠ eval_loss much lower than train_loss (-2σ; check eval split overlap)"
    if gap > 1.0:
        return "⚠ severe overfit (eval - train > 1.0)"
    if gap > 0.5:
        return "⚠ mild overfit"
    return "✓"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["banking77", "bitext_support", "cuad"])
    parser.add_argument("--method", required=True,
                        choices=list(SUPPORTED_METHODS))
    parser.add_argument("--scale", required=True, choices=["pilot", "larger"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    run_one(args.task, args.method, args.scale, args.seed, args.model_path)


if __name__ == "__main__":
    main()
