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
    }
    append_jsonl(record, ALL_RUNS_PATH)

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    logger.info(f"[{task}/{scale}/{method}] DONE. eval_quality={eval_quality_raw}")
    return record


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
