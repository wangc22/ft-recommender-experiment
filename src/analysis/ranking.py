"""Ranking consistency between pilot and larger runs over PEFT methods."""
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau


def _ranks_in_method_order(recs, score_key: str, method_order: list[str]) -> list[int]:
    """Return ranks (rank=1 means best) for each method in method_order."""
    sorted_ = sorted(recs, key=lambda r: -r[score_key])
    rank_of = {r["method"]: i + 1 for i, r in enumerate(sorted_)}
    return [rank_of.get(m, len(method_order)) for m in method_order]


def analyze_pilot_vs_larger(records: list[dict]) -> dict:
    """records: scored runs (composite_score filled). Returns per-task analysis."""
    by_task = defaultdict(lambda: {"pilot": [], "larger": []})
    for r in records:
        by_task[r["task"]][r["run_type"]].append(r)

    out = {}
    for task, splits in by_task.items():
        pilot = splits["pilot"]
        larger = splits["larger"]
        method_set = sorted({r["method"] for r in pilot} | {r["method"] for r in larger})
        if not method_set or len(pilot) != len(method_set) or len(larger) != len(method_set):
            out[task] = {"error": f"missing runs: pilot={len(pilot)} larger={len(larger)} "
                                   f"methods_seen={method_set}"}
            continue

        pilot_ranks = _ranks_in_method_order(pilot, "composite_score", method_set)
        larger_ranks = _ranks_in_method_order(larger, "eval_quality_raw", method_set)

        sp = spearmanr(pilot_ranks, larger_ranks)
        kt = kendalltau(pilot_ranks, larger_ranks)

        pilot_top1 = max(pilot, key=lambda r: r["composite_score"])["method"]
        larger_top1 = max(larger, key=lambda r: r["eval_quality_raw"])["method"]

        pilot_top2 = sorted(pilot, key=lambda r: -r["composite_score"])[:2]
        pilot_top2_methods = [r["method"] for r in pilot_top2]

        larger_by_method = {r["method"]: r for r in larger}
        larger_best = max(r["eval_quality_raw"] for r in larger)
        pilot_pick_q_in_larger = larger_by_method[pilot_top1]["eval_quality_raw"]
        regret_abs = larger_best - pilot_pick_q_in_larger
        regret_rel = regret_abs / larger_best if larger_best > 1e-9 else 0.0

        sp_val = float(sp.statistic) if sp.statistic == sp.statistic else None
        kt_val = float(kt.statistic) if kt.statistic == kt.statistic else None

        out[task] = {
            "method_set": method_set,
            "pilot_top1": pilot_top1,
            "pilot_top2": pilot_top2_methods,
            "larger_top1": larger_top1,
            "top1_match": pilot_top1 == larger_top1,
            "top2_overlap": larger_top1 in pilot_top2_methods,
            "regret_absolute": round(regret_abs, 4),
            "regret_relative": round(regret_rel, 4),
            "spearman": round(sp_val, 4) if sp_val is not None else None,
            "kendall_tau": round(kt_val, 4) if kt_val is not None else None,
            "pilot_ranks": dict(zip(method_set, pilot_ranks)),
            "larger_ranks": dict(zip(method_set, larger_ranks)),
            "success": (
                pilot_top1 == larger_top1
                or larger_top1 in pilot_top2_methods
                or regret_rel <= 0.02
            ),
        }
    return out
