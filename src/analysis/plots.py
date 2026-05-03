"""Key plots for the experiment report."""
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("outputs/figures")

METHOD_COLOR = {
    "lora": "tab:blue",
    "dora": "tab:orange",
    "qlora": "tab:green",
    "lora_plus": "tab:red",
}


def _ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_rank_consistency(ranking_analysis: dict):
    _ensure_out()
    tasks = list(ranking_analysis.keys())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        info = ranking_analysis[task]
        if "error" in info:
            ax.set_title(f"{task} (error)")
            continue
        methods = info["method_set"]
        x = np.arange(len(methods))
        width = 0.35
        pilot_r = [info["pilot_ranks"][m] for m in methods]
        larger_r = [info["larger_ranks"][m] for m in methods]
        ax.bar(x - width / 2, pilot_r, width, label="Pilot")
        ax.bar(x + width / 2, larger_r, width, label="Larger")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20)
        ax.set_ylabel("Rank (1 = best)")
        ax.set_title(f"{task}\ntop1_match={info['top1_match']} regret={info['regret_relative']:.2%}")
        ax.invert_yaxis()
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rank_consistency.png", dpi=150)
    plt.close()


def plot_cost_perf_scatter(scored: list[dict]):
    _ensure_out()
    by_task = defaultdict(list)
    for r in scored:
        by_task[r["task"]].append(r)
    fig, axes = plt.subplots(1, len(by_task), figsize=(5 * len(by_task), 4))
    if len(by_task) == 1:
        axes = [axes]
    scale_marker = {"pilot": "o", "larger": "s"}
    for ax, (task, recs) in zip(axes, by_task.items()):
        for r in recs:
            ax.scatter(r["training_time"], r["eval_quality_raw"],
                       c=METHOD_COLOR.get(r["method"], "gray"),
                       marker=scale_marker[r["run_type"]],
                       s=120, edgecolor="black",
                       label=f"{r['method']} {r['run_type']}")
        ax.set_xlabel("Training time (s)")
        ax.set_ylabel("Eval quality (raw)")
        ax.set_title(task)
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        ax.legend(seen.values(), seen.keys(), fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cost_perf_scatter.png", dpi=150)
    plt.close()


def plot_pilot_composite_bars(scored: list[dict]):
    _ensure_out()
    by_task = defaultdict(dict)
    methods_seen = set()
    for r in scored:
        if r["run_type"] != "pilot":
            continue
        by_task[r["task"]][r["method"]] = r["composite_score"]
        methods_seen.add(r["method"])
    methods = sorted(methods_seen)
    fig, ax = plt.subplots(figsize=(8, 4))
    tasks = list(by_task.keys())
    x = np.arange(len(tasks))
    width = 0.8 / max(len(methods), 1)
    for i, method in enumerate(methods):
        vals = [by_task[t].get(method, 0) for t in tasks]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width,
               label=method, color=METHOD_COLOR.get(method, None))
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Pilot composite score")
    ax.set_title("Pilot composite score by task and PEFT method")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pilot_composite_bars.png", dpi=150)
    plt.close()


def plot_memory_quality(scored: list[dict]):
    """Highlights QLoRA's memory advantage vs other methods on larger runs."""
    _ensure_out()
    by_task = defaultdict(list)
    for r in scored:
        if r["run_type"] == "larger":
            by_task[r["task"]].append(r)
    fig, axes = plt.subplots(1, len(by_task), figsize=(5 * len(by_task), 4))
    if len(by_task) == 1:
        axes = [axes]
    for ax, (task, recs) in zip(axes, by_task.items()):
        for r in recs:
            ax.scatter(r["memory_cost"], r["eval_quality_raw"],
                       c=METHOD_COLOR.get(r["method"], "gray"),
                       s=180, edgecolor="black", label=r["method"])
            ax.annotate(r["method"], (r["memory_cost"], r["eval_quality_raw"]),
                        fontsize=8, xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("Memory cost (MB, larger run)")
        ax.set_ylabel("Eval quality")
        ax.set_title(f"{task} — memory vs quality")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "memory_quality.png", dpi=150)
    plt.close()


def plot_recommendation_summary(ranking_analysis: dict):
    _ensure_out()
    tasks = list(ranking_analysis.keys())
    rows = []
    for t in tasks:
        info = ranking_analysis[t]
        if "error" in info:
            rows.append([t, "—", "—", "—", "ERR"])
            continue
        rows.append([
            t,
            info["pilot_top1"],
            info["larger_top1"],
            f"{info['regret_relative']:.2%}",
            "PASS" if info["success"] else "FAIL",
        ])
    fig, ax = plt.subplots(figsize=(9, 1 + 0.5 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Task", "Pilot top-1", "Larger top-1", "Regret", "Decision"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    plt.title("Final Recommendation Summary", pad=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "recommendation_summary.png", dpi=150)
    plt.close()


def make_all(scored: list[dict], ranking_analysis: dict):
    plot_rank_consistency(ranking_analysis)
    plot_cost_perf_scatter(scored)
    plot_pilot_composite_bars(scored)
    plot_memory_quality(scored)
    plot_recommendation_summary(ranking_analysis)
