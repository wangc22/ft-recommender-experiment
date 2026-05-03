"""Generation evaluation: greedy decode + ROUGE/BLEU + format checker + LLM judge."""
import re
import torch

from src.data.formatters import format_generation_prompt_only
from src.evaluation.llm_judge import judge_batch
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _format_correct(reply: str) -> bool:
    """Heuristic format checks: non-empty, no stray system-prompt leakage,
    no repeated instruction text, ends without trailing user/assistant tags."""
    if not reply or len(reply.strip()) < 5:
        return False
    if re.search(r"<\|im_start\|>|<\|im_end\|>|<<SYS>>", reply):
        return False
    if reply.count("Assistant:") > 1 or reply.count("Customer:") > 0:
        return False
    return True


def predict_batch(model, tokenizer, prompts: list[str],
                  batch_size: int = 4, max_new_tokens: int = 256) -> list[str]:
    model.eval()
    device = next(model.parameters()).device
    outs = []
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
        for j in range(len(batch)):
            input_len = enc["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
            full = tokenizer.decode(gen[j], skip_special_tokens=True)
            inp_dec = tokenizer.decode(enc["input_ids"][j][:input_len],
                                       skip_special_tokens=True)
            if full.startswith(inp_dec):
                outs.append(full[len(inp_dec):].strip())
            else:
                outs.append(tokenizer.decode(gen[j][enc["input_ids"].shape[1]:],
                                             skip_special_tokens=True).strip())
    return outs


def _compute_rouge_bleu(predictions: list[str], references: list[str]) -> dict:
    metrics = {}
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(r, p)["rougeL"].fmeasure
                  for p, r in zip(predictions, references)]
        metrics["rougeL"] = round(sum(scores) / len(scores), 4) if scores else 0.0
    except Exception as e:
        logger.warning(f"ROUGE failed: {e}")
        metrics["rougeL"] = None
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        metrics["bleu"] = round(bleu.score, 4)
    except Exception as e:
        logger.warning(f"BLEU failed: {e}")
        metrics["bleu"] = None
    return metrics


def evaluate_generation(model, tokenizer, records: list[dict]) -> dict:
    user_queries = [r["messages"][0]["content"] for r in records]
    references = [r["messages"][-1]["content"] for r in records]
    prompts = [format_generation_prompt_only(q, tokenizer) for q in user_queries]
    predictions = predict_batch(model, tokenizer, prompts)

    format_pass = sum(1 for p in predictions if _format_correct(p))
    format_pass_rate = format_pass / len(predictions) if predictions else 0.0

    word_metrics = _compute_rouge_bleu(predictions, references)

    judge_result = judge_batch(user_queries, predictions)
    judge_means = judge_result["means"]

    eval_quality = round(0.7 * (judge_means["judge_avg"] / 5.0)
                         + 0.3 * format_pass_rate, 4)

    return {
        "eval_quality": eval_quality,
        "judge_avg": judge_means["judge_avg"],
        "judge_helpfulness": judge_means["helpfulness"],
        "judge_format": judge_means["format"],
        "judge_policy_consistency": judge_means["policy_consistency"],
        "format_pass_rate": round(format_pass_rate, 4),
        "rougeL": word_metrics.get("rougeL"),
        "bleu": word_metrics.get("bleu"),
        "n_valid_judge": judge_means["n_valid"],
        "n_total": len(records),
        "predictions": [
            {"user": user_queries[i][:200], "reference": references[i][:200],
             "prediction": predictions[i][:300],
             "judge": judge_result["per_sample"][i]["scores"]}
            for i in range(len(records))
        ],
    }
