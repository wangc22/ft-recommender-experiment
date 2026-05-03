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
                  batch_size: int = 8, max_new_tokens: int = 32) -> list[str]:
    model.eval()
    device = next(model.parameters()).device
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=1024, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j, prompt in enumerate(batch):
            input_len = enc["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
            full_decoded = tokenizer.decode(gen[j], skip_special_tokens=True)
            decoded_input = tokenizer.decode(enc["input_ids"][j][:input_len],
                                             skip_special_tokens=True)
            if full_decoded.startswith(decoded_input):
                completion = full_decoded[len(decoded_input):]
            else:
                completion = tokenizer.decode(gen[j][enc["input_ids"].shape[1]:],
                                              skip_special_tokens=True)
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
