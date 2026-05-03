"""Within-task min-max normalization + composite score."""
from collections import defaultdict


DEFAULT_WEIGHTS = {
    "quality": 0.45,
    "training_time": 0.20,
    "memory": 0.25,
    "overfit": 0.10,
}


def _normalize(values: list[float], higher_is_better: bool) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1e-9:
        return [0.5 for _ in values]
    norm = [(v - lo) / span for v in values]
    if not higher_is_better:
        norm = [1.0 - x for x in norm]
    return norm


def compute_composite(records: list[dict], weights: dict = None) -> list[dict]:
    """Mutates a copy of records: adds composite_score (within-task min-max).

    Higher composite_score = better trade-off.
    """
    weights = weights or DEFAULT_WEIGHTS
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)

    out = []
    for task, recs in by_task.items():
        qualities = [r["eval_quality_raw"] for r in recs]
        times = [r["training_time"] for r in recs]
        mems = [r["memory_cost"] for r in recs]
        gaps = [r["overfit_gap"] for r in recs]

        n_q = _normalize(qualities, higher_is_better=True)
        n_t = _normalize(times, higher_is_better=False)
        n_m = _normalize(mems, higher_is_better=False)
        n_o = _normalize(gaps, higher_is_better=False)

        for i, r in enumerate(recs):
            r2 = dict(r)
            r2["composite_score"] = round(
                weights["quality"] * n_q[i]
                + weights["training_time"] * n_t[i]
                + weights["memory"] * n_m[i]
                + weights["overfit"] * n_o[i],
                4,
            )
            r2["_normalized"] = {
                "quality": round(n_q[i], 4),
                "training_time": round(n_t[i], 4),
                "memory": round(n_m[i], 4),
                "overfit": round(n_o[i], 4),
            }
            out.append(r2)
    return out
