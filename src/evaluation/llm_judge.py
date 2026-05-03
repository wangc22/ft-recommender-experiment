"""Anthropic Claude API judge for customer-support generation outputs.

Fixed rubric: helpfulness / format / policy_consistency, each 1-5.
Returns a per-sample dict + aggregate means.
"""
import json
import os
import re
import time

from src.utils.logging import get_logger

logger = get_logger(__name__)

JUDGE_MODEL = "claude-haiku-4-5-20251001"

JUDGE_SYSTEM = """You are a strict evaluator of customer-support assistant replies.
Score on three dimensions, integer 1-5 each:

- helpfulness: does it directly address the customer's request and propose useful next steps?
- format: is it well-structured, polite, concise, and free of stray instruction text?
- policy_consistency: does it avoid promising unauthorized refunds, breaking confidentiality,
  or contradicting standard customer-support policy?

Return ONLY a JSON object: {"helpfulness": int, "format": int, "policy_consistency": int, "rationale": "<one short sentence>"}.
"""

JUDGE_USER_TEMPLATE = """Customer: {user}

Assistant reply: {reply}

Score the reply."""


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge_one(client, user_query: str, reply: str, max_retries: int = 3) -> dict | None:
    user_msg = JUDGE_USER_TEMPLATE.format(user=user_query[:2000], reply=reply[:3000])
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=300,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = "".join(b.text for b in resp.content if hasattr(b, "text"))
            parsed = _extract_json(text)
            if parsed and all(k in parsed for k in ("helpfulness", "format", "policy_consistency")):
                return parsed
            last_err = f"unparseable: {text[:200]}"
        except Exception as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    logger.warning(f"Judge failed for one sample after {max_retries} retries: {last_err}")
    return None


def judge_batch(user_queries: list[str], replies: list[str]) -> dict:
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Generation evaluation requires the LLM judge; "
            "set the env var before running."
        )
    from anthropic import Anthropic
    client = Anthropic()

    per_sample = []
    sums = {"helpfulness": 0.0, "format": 0.0, "policy_consistency": 0.0}
    valid = 0
    for q, r in zip(user_queries, replies):
        scored = judge_one(client, q, r)
        if scored:
            for k in sums:
                sums[k] += scored[k]
            valid += 1
        per_sample.append({"user": q[:200], "reply": r[:200], "scores": scored})

    if valid == 0:
        raise RuntimeError("Judge returned no valid scores for any sample.")

    means = {k: round(sums[k] / valid, 3) for k in sums}
    means["judge_avg"] = round(sum(means.values()) / 3, 3)
    means["n_valid"] = valid
    means["n_total"] = len(replies)
    return {"means": means, "per_sample": per_sample}
