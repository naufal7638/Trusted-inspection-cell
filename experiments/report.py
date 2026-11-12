"""Turn predictions.jsonl into report.md and a chart."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import BAD_IMAGE_LABELS

DECISIONS = ("PASS", "REJECT", "REVIEW")
METHOD_TITLES = {
    "trained_classifier": "Baseline 1: trained classifier",
    "naive_vlm": "Baseline 2: naive Gemma (whole image)",
    "inspection_cell": "Ours: inspection cell",
}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}% ({n}/{d})" if d else "-"


def metrics(rows: list[dict]) -> dict:
    good = [r for r in rows if r["label"] == "good"]
    defects = [r for r in rows if r["label"] not in ("good", *BAD_IMAGE_LABELS)]
    count = lambda rs, d: sum(1 for r in rs if r["decision"] == d)
    latencies = [r["latency_s"] for r in rows]
    return {
        "n": len(rows),
        "accuracy": (sum(r["decision"] == r["expected"] for r in rows), len(rows)),
        "escaped": (count(defects, "PASS"), len(defects)),
        "caught_auto": (count(defects, "REJECT"), len(defects)),
        "good_passed": (count(good, "PASS"), len(good)),
        "false_reject": (count(good, "REJECT"), len(good)),
        "review": (count(rows, "REVIEW"), len(rows)),
        "latency": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def write_report(out_dir: Path, info: dict) -> Path:
    rows = load(out_dir / "predictions.jsonl")
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    methods = [m for m in METHOD_TITLES if m in by_method]

    # Compare everything on the images all methods were run on, so numbers are like-for-like.
    shared = set.intersection(*(set(r["image"] for r in by_method[m]) for m in methods)) if methods else set()

    lines = [f"# Performance report: {info['name']}", ""]
    lines += [
        "## Setup",
        "",
        f"- Training set: `{info['train']}` ({info['n_train']} images)",
        f"- Test set: `{info['test']}` ({info['n_test']} images); Gemma methods run on a stratified subset",
        f"- Model: {info['model']} via Ollama, CPU only" + (" (MOCK model: dry run, numbers are meaningless)" if info.get("mock") else ""),
        f"- Images every method was tested on: {len(shared)}",
        "",
        "Expected decision: good -> PASS, any defect -> REJECT, unusable image -> REVIEW.",
        "**Escaped defects** (defect decided PASS) is the number that matters most: those parts ship.",
        "A REVIEW on a defective part is safe (a human sees it) but costs operator time.",
        "",
        f"## Results on the shared test images (n = {len(shared)})",
        "",
        "| Method | Accuracy | Escaped defects ↓ | Defects auto-rejected | Good parts passed | Good parts falsely rejected ↓ | Sent to review | Time / part |",
        "|---|---|---|---|---|---|---|---|",
    ]
    summary = {}
    for m in methods:
        s = metrics([r for r in by_method[m] if r["image"] in shared])
        summary[m] = s
        lines.append(
            f"| {METHOD_TITLES[m]} | {pct(*s['accuracy'])} | {pct(*s['escaped'])} | {pct(*s['caught_auto'])} | "
            f"{pct(*s['good_passed'])} | {pct(*s['false_reject'])} | {pct(*s['review'])} | {s['latency']:.1f}s |"
        )

    for m in methods:
        if len(by_method[m]) > len(shared):
            s = metrics(by_method[m])
            lines += ["", f"{METHOD_TITLES[m]} on its full test set (n = {s['n']}): accuracy {pct(*s['accuracy'])}, "
                      f"escaped defects {pct(*s['escaped'])}, good parts falsely rejected {pct(*s['false_reject'])}."]

    lines += ["", "## Decisions by true label (shared test images)", ""]
    for m in methods:
        table = defaultdict(Counter)
        for r in by_method[m]:
            if r["image"] in shared:
                table[r["label"]][r["decision"]] += 1
        lines += [f"**{METHOD_TITLES[m]}**", "", "| True label | " + " | ".join(DECISIONS) + " |", "|---" * (len(DECISIONS) + 1) + "|"]
        for label in sorted(table):
            lines.append(f"| {label} | " + " | ".join(str(table[label][d]) for d in DECISIONS) + " |")
        lines.append("")

    chart = out_dir / "comparison.png"
    _chart(summary, chart)
    lines += [f"![comparison]({chart.name})", ""]

    report = out_dir / "report.md"
    report.write_text("\n".join(lines))
    return report


def _chart(summary: dict, path: Path) -> None:
    if not summary:
        return
    bars = [("Accuracy", "accuracy"), ("Escaped defects", "escaped"), ("False rejects", "false_reject"), ("Sent to review", "review")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(summary)
    colours = ["#9ca3af", "#f59e0b", "#2563eb"]
    for i, (m, s) in enumerate(summary.items()):
        vals = [100 * s[k][0] / s[k][1] if s[k][1] else 0 for _, k in bars]
        xs = [j + i * width for j in range(len(bars))]
        ax.bar(xs, vals, width, label=METHOD_TITLES[m], color=colours[i % len(colours)])
        for x, v in zip(xs, vals):
            ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks([j + width * (len(summary) - 1) / 2 for j in range(len(bars))], [b for b, _ in bars])
    ax.set_ylabel("% of test images")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
