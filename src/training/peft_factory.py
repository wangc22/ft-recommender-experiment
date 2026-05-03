"""PEFT method factory: returns (lora_config, model_kwargs, optimizer_factory) per method.

Supported methods (case-insensitive):
- lora      : standard LoRA, r=8 alpha=16
- dora      : LoRA with use_dora=True (DoRA), same r/alpha
- qlora     : 4-bit nf4 quantized base model + LoRA on top, same r/alpha
- lora_plus : standard LoRA + custom optimizer with B-matrix LR ratio = 16
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from peft import LoraConfig, TaskType

TARGET_MODULE_MAP = {
    "qwen2":   ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen":    ["c_attn", "c_proj"],
    "llama":   ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
    "gemma":   ["q_proj", "v_proj"],
    "default": ["q_proj", "v_proj"],
}

DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
LORA_PLUS_LR_RATIO = 16.0

SUPPORTED_METHODS = ("lora", "dora", "qlora", "lora_plus")


@dataclass
class PeftPlan:
    method: str
    lora_config: LoraConfig
    # extra kwargs forwarded to AutoModelForCausalLM.from_pretrained (e.g. quantization_config)
    model_kwargs: dict = field(default_factory=dict)
    # if not None, called after trainer is built to override its optimizer
    # signature: (peft_model, base_lr) -> torch.optim.Optimizer
    optimizer_factory: Optional[Callable] = None
    notes: str = ""


def _target_modules_for(model_config) -> list[str]:
    arch = getattr(model_config, "model_type", "default").lower()
    return TARGET_MODULE_MAP.get(arch, TARGET_MODULE_MAP["default"])


def _base_lora(model_config, use_dora: bool = False) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=DEFAULT_LORA_R,
        lora_alpha=DEFAULT_LORA_ALPHA,
        lora_dropout=DEFAULT_LORA_DROPOUT,
        target_modules=_target_modules_for(model_config),
        bias="none",
        use_dora=use_dora,
    )


def _qlora_quant_kwargs(compute_dtype) -> dict:
    from transformers import BitsAndBytesConfig
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        ),
    }


def _lora_plus_optimizer_factory(peft_model, base_lr: float):
    """Wrap AdamW with LoRA+ B/A LR split."""
    from peft.optimizers import create_loraplus_optimizer
    return create_loraplus_optimizer(
        model=peft_model,
        optimizer_cls=torch.optim.AdamW,
        lr=base_lr,
        loraplus_lr_ratio=LORA_PLUS_LR_RATIO,
    )


def build_peft(method: str, model_config, compute_dtype=torch.bfloat16) -> PeftPlan:
    m = method.lower()
    if m == "lora":
        return PeftPlan(method=m, lora_config=_base_lora(model_config))
    if m == "dora":
        return PeftPlan(method=m, lora_config=_base_lora(model_config, use_dora=True),
                        notes="DoRA: magnitude/direction decomposition")
    if m == "qlora":
        return PeftPlan(
            method=m,
            lora_config=_base_lora(model_config),
            model_kwargs=_qlora_quant_kwargs(compute_dtype),
            notes="QLoRA: 4-bit nf4 base + LoRA on top",
        )
    if m == "lora_plus":
        return PeftPlan(
            method=m,
            lora_config=_base_lora(model_config),
            optimizer_factory=_lora_plus_optimizer_factory,
            notes=f"LoRA+: B-matrix lr ratio = {LORA_PLUS_LR_RATIO}",
        )
    raise ValueError(f"Unknown PEFT method '{method}'. Supported: {SUPPORTED_METHODS}")


def count_trainable_params(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio_pct": round(100 * trainable / total, 4),
    }
