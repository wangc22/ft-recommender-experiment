"""Classification evaluation: greedy decode + label parsing + accuracy/macro-F1."""
import re
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.data.formatters import format_classification_prompt
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_label(prediction_text: str, valid_labels: list[str]) -> str | None:
    """Find a valid label in the model output. Match whole label first, then case-insensitive."""
    pred = prediction_text.strip()
    pred_lower = pred.lower()
    sorted_labels = sorted(valid_labels, key=lambda x: -len(x))
    for label in sorted_labels:
        if label.lower() in pred_lower:
            if re.search(rf"\b{re.escape(label.lower())}\b", pred_lower):
                return label
    for label in sorted_labels:
        if pred_lower.startswith(label.lower()):
            return label
    return None


def predict_batch(model, tokenizer, prompts: list[str],
                  batch_size: int = 8, max_new_tokens: int = 32,
                  max_input_length: int = 2048) -> list[str]:
    """Greedy-decode each prompt; return only the generated continuation.

    Uses token-level slicing at the (left-padded) prompt boundary — robust to
    variable prompt lengths within a batch. The earlier string-based
    `full.startswith(decoded_input)` heuristic was fragile for short samples
    in the same batch (decoded_input would resolve to all-pad → empty string,
    causing the entire prompt+completion to be returned and the label parser
    matching labels embedded in the prompt's label list).
    """
    model.eval()
    device = next(model.parameters()).device
    # QLoRA path needs autocast at inference: prepare_model_for_kbit_training
    # upcasts lm_head to fp32 for grad stability, but bnb 4-bit modules emit
    # bf16 hidden states. Without autocast: 'expected scalar type Float but
    # found BFloat16'. Harmless for non-QLoRA models (autocast is a no-op when
    # weight already matches input dtype).
    use_autocast = torch.cuda.is_available() and device.type == "cuda"
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_input_length, return_tensors="pt").to(device)
        prompt_len = enc["input_ids"].shape[1]  # padded length; same for all rows
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast,
        ):
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j in range(len(batch)):
            completion_ids = gen[j][prompt_len:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            outputs.append(completion.strip())
    return outputs


def evaluate_classification(model, tokenizer, records: list[dict],
                            labels: list[str]) -> dict:
    prompts = [format_classification_prompt(r["input"], labels) for r in records]
    predictions = predict_batch(model, tokenizer, prompts)

    parsed = []
    n_invalid = 0
    for p in predictions:
        label = parse_label(p, labels)
        if label is None:
            n_invalid += 1
            label = "<invalid>"
        parsed.append(label)

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    label_to_idx["<invalid>"] = -1

    y_true = [label_to_idx[r["label"]] for r in records]
    y_pred = [label_to_idx[p] for p in parsed]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=list(range(len(labels))),
                         average="macro", zero_division=0)

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "n_invalid": n_invalid,
        "n_total": len(records),
        "predictions": [
            {"input": records[i]["input"][:200], "true": records[i]["label"],
             "raw": predictions[i], "parsed": parsed[i]}
            for i in range(len(records))
        ],
    }
