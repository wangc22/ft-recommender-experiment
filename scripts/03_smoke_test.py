"""Smoke test: minimal data + 2 steps to verify each (task, method) train+eval path.

Validates wiring (data formatter, trl trainer, peft method, predict, label parser, judge).
On Mac MPS, qlora is skipped because bitsandbytes 4-bit kernels do not work on Apple Silicon.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import Dataset
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.data.formatters import (
    format_classification_for_sft,
    format_generation_for_sft,
    format_generation_prompt_only,
)
from src.evaluation.classification import evaluate_classification
from src.evaluation.generation import _format_correct, predict_batch
from src.training.peft_factory import (
    SUPPORTED_METHODS,
    build_peft,
    count_trainable_params,
)
from src.utils.io import load_json, load_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("smoke")

MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"
N_TRAIN = 8
N_VAL = 4


def _device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def _make_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def _build_model_with_peft(method: str, device_kind: str, dtype):
    plan = build_peft(method, AutoModelForCausalLM.from_pretrained(MODEL_PATH).config,
                      compute_dtype=dtype)
    kwargs = {"torch_dtype": dtype, **plan.model_kwargs}
    if device_kind == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    if device_kind == "mps" and "device_map" not in kwargs:
        model = model.to("mps")
    if method == "qlora":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, plan.lora_config)
    return model, plan


def smoke_classification(task: str, method: str):
    logger.info(f"--- smoke {task}/{method} (classification) ---")
    set_seed(42)
    base_dir = Path("data/processed") / task
    labels = load_json(str(base_dir / "labels.json"))
    train_recs = load_jsonl(str(base_dir / "pilot_train.jsonl"))[:N_TRAIN]
    val_recs = load_jsonl(str(base_dir / "val.jsonl"))[:N_VAL]

    tokenizer = _make_tokenizer()
    device_kind, dtype = _device_and_dtype()
    model, plan = _build_model_with_peft(method, device_kind, dtype)
    logger.info(f"Trainable: {count_trainable_params(model)} | {plan.notes or '-'}")

    train_ds = Dataset.from_list([format_classification_for_sft(r, labels) for r in train_recs])
    val_ds = Dataset.from_list([format_classification_for_sft(r, labels) for r in val_recs])

    args = SFTConfig(
        output_dir=f"outputs/smoke/{task}_{method}",
        max_steps=2, per_device_train_batch_size=2,
        per_device_eval_batch_size=2, gradient_accumulation_steps=1,
        learning_rate=2e-4, max_length=256,
        eval_strategy="no", save_strategy="no",
        logging_steps=1, report_to="none",
    )
    optimizers = (None, None)
    if plan.optimizer_factory is not None:
        optimizers = (plan.optimizer_factory(model, 2e-4), None)
        logger.info("LoRA+ optimizer injected")
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds,
                         eval_dataset=val_ds, optimizers=optimizers)
    trainer.train()
    logger.info("training OK; running classification eval (3 samples)...")
    metrics = evaluate_classification(trainer.model, tokenizer, val_recs[:3], labels)
    logger.info(f"eval ok: acc={metrics['accuracy']} macro_f1={metrics['macro_f1']} "
                f"n_invalid={metrics['n_invalid']}")


def smoke_generation(method: str):
    logger.info(f"--- smoke bitext_support/{method} (generation) ---")
    set_seed(42)
    base_dir = Path("data/processed/bitext_support")
    train_recs = load_jsonl(str(base_dir / "pilot_train.jsonl"))[:N_TRAIN]
    val_recs = load_jsonl(str(base_dir / "val.jsonl"))[:N_VAL]

    tokenizer = _make_tokenizer()
    device_kind, dtype = _device_and_dtype()
    model, plan = _build_model_with_peft(method, device_kind, dtype)

    train_ds = Dataset.from_list([format_generation_for_sft(r) for r in train_recs])

    args = SFTConfig(
        output_dir=f"outputs/smoke/bitext_{method}",
        max_steps=2, per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=2e-4, max_length=512,
        eval_strategy="no", save_strategy="no",
        logging_steps=1, report_to="none",
    )
    optimizers = (None, None)
    if plan.optimizer_factory is not None:
        optimizers = (plan.optimizer_factory(model, 2e-4), None)
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds,
                         optimizers=optimizers)
    trainer.train()

    logger.info("training OK; running greedy decode + format check...")
    prompts = [format_generation_prompt_only(r["messages"][0]["content"], tokenizer)
               for r in val_recs[:2]]
    preds = predict_batch(trainer.model, tokenizer, prompts, batch_size=1, max_new_tokens=64)
    pass_n = sum(1 for p in preds if _format_correct(p))
    logger.info(f"got {len(preds)} predictions, format_pass={pass_n}/{len(preds)}, "
                f"sample={preds[0][:120]!r}")


def main():
    device_kind, _ = _device_and_dtype()
    logger.info(f"=== device: {device_kind} ===")

    methods_to_test = list(SUPPORTED_METHODS)
    if device_kind != "cuda":
        methods_to_test = [m for m in methods_to_test if m != "qlora"]
        logger.warning("Skipping qlora on non-CUDA device (bitsandbytes 4-bit only on CUDA).")

    for method in methods_to_test:
        try:
            smoke_classification("banking77", method)
        except Exception as e:
            logger.error(f"FAIL banking77/{method}: {e}")
            raise
        try:
            smoke_classification("cuad", method)
        except Exception as e:
            logger.error(f"FAIL cuad/{method}: {e}")
            raise
        try:
            smoke_generation(method)
        except Exception as e:
            logger.error(f"FAIL bitext/{method}: {e}")
            raise

    if os.getenv("ANTHROPIC_API_KEY"):
        from src.evaluation.llm_judge import judge_batch
        sample_user = "Where is my refund?"
        sample_reply = "Let me check the status of your refund. Could you share the order ID?"
        res = judge_batch([sample_user], [sample_reply])
        logger.info(f"judge OK: means={res['means']}")
    else:
        logger.warning("ANTHROPIC_API_KEY not set — skipping judge smoke. "
                       "Set it on cloud before run_matrix.")
    logger.info("=== ALL SMOKE TESTS PASSED ===")


if __name__ == "__main__":
    main()
