"""Generation evaluation.

Two phases (separable, by design):

1. **Predict** (cloud-side, no API key needed):
   greedy decode + ROUGE-L + BLEU + format check. Saves predictions to
   eval_metrics.json with `judge_pending=True`.

2. **Judge** (local-side, post-hoc):
   reads predictions, calls Anthropic LLM-as-Judge, writes back judge_avg /
   eval_quality. See scripts/run_judge.py.

This split keeps cloud runs free of API keys and faster (no per-sample API
latency). Re-judging with a different rubric can be done locally without
retraining.
"""
import os
import re
import torch

from src.data.formatters import format_generation_prompt_only
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
                  batch_size: int = 4, max_new_tokens: int = 256,
                  max_input_length: int = 2048) -> list[str]:
    """Token-level slicing at the padded prompt boundary (robust to left padding)."""
    model.eval()
    device = next(model.parameters()).device
    # See classification.predict_batch for why autocast is needed (QLoRA path).
    use_autocast = torch.cuda.is_available() and device.type == "cuda"
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_input_length, return_tensors="pt").to(device)
        prompt_len = enc["input_ids"].shape[1]
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
            outs.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())
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


def evaluate_generation(model, tokenizer, records: list[dict],
                         skip_judge: bool | None = None) -> dict:
    """Evaluate a generation model.

    `skip_judge`:
      - None (default): controlled by env var `LLM_JUDGE`. `LLM_JUDGE=skip` →
        skip; otherwise run judge if `ANTHROPIC_API_KEY` is in env.
      - True: never call judge; mark judge_pending=True
      - False: always call judge (raises if key missing)

    When judge is skipped:
      - eval_quality is set to a heuristic proxy (ROUGE-L * 0.5 + format_pass * 0.5)
      - judge_pending = True is set so post-hoc scripts/run_judge.py can fill in
        the real eval_quality later.
    """
    user_queries = [r["messages"][0]["content"] for r in records]
    references = [r["messages"][-1]["content"] for r in records]
    prompts = [format_generation_prompt_only(q, tokenizer) for q in user_queries]
    predictions = predict_batch(model, tokenizer, prompts)

    format_pass = sum(1 for p in predictions if _format_correct(p))
    format_pass_rate = format_pass / len(predictions) if predictions else 0.0
    word_metrics = _compute_rouge_bleu(predictions, references)

    # Decide whether to run judge
    if skip_judge is None:
        env_mode = os.environ.get("LLM_JUDGE", "auto").lower()
        if env_mode == "skip":
            skip_judge = True
        else:
            # auto: skip if no API key
            skip_judge = "ANTHROPIC_API_KEY" not in os.environ

    if skip_judge:
        # Heuristic proxy quality. Real eval_quality filled in by run_judge.py later.
        rouge = word_metrics.get("rougeL") or 0.0
        proxy_quality = round(0.5 * rouge + 0.5 * format_pass_rate, 4)
        logger.info(
            f"LLM judge SKIPPED (proxy eval_quality={proxy_quality}); "
            "run scripts/run_judge.py locally to fill in real judge scores."
        )
        return {
            "eval_quality": proxy_quality,
            "judge_pending": True,
            "judge_avg": None,
            "judge_helpfulness": None,
            "judge_format": None,
            "judge_policy_consistency": None,
            "format_pass_rate": round(format_pass_rate, 4),
            "rougeL": word_metrics.get("rougeL"),
            "bleu": word_metrics.get("bleu"),
            "n_valid_judge": 0,
            "n_total": len(records),
            "predictions": [
                {"user": user_queries[i], "reference": references[i],
                 "prediction": predictions[i],
                 "judge": None}
                for i in range(len(records))
            ],
        }

    # Inline judge path (rarely used now; preserved for backward compat)
    from src.evaluation.llm_judge import judge_batch
    judge_result = judge_batch(user_queries, predictions)
    judge_means = judge_result["means"]
    eval_quality = round(0.7 * (judge_means["judge_avg"] / 5.0)
                         + 0.3 * format_pass_rate, 4)
    return {
        "eval_quality": eval_quality,
        "judge_pending": False,
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
            {"user": user_queries[i], "reference": references[i],
             "prediction": predictions[i],
             "judge": judge_result["per_sample"][i]["scores"]}
            for i in range(len(records))
        ],
    }
