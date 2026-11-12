"""Run the baseline vs inspection-cell experiment and write a performance report.

    python experiment.py --name synthetic --train data/synthetic/train --test data/synthetic/test
    python experiment.py --name real --train photos/train --test photos/test
    python experiment.py --name synthetic --report-only

Gemma is slow on CPU, so the two Gemma methods run on `--vlm-per-label` images per label
(interleaved across labels, so a run stopped early is still balanced). Results are saved
after every image; re-running the same command resumes where it stopped.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from dataclasses import replace

from cell.config import PART_QUALITY, CellConfig
from cell.presence import calibrate
from cell.spec import SPECS
from experiments.dataset import expected_decision, label_of, list_images
from experiments.methods import InspectionCell, NaiveVLM, TrainedClassifier, timed
from experiments.report import write_report


def stratified(images: list[Path], per_label: int) -> list[Path]:
    groups = defaultdict(list)
    for p in images:
        groups[label_of(p)].append(p)
    picked = [groups[label][:per_label] for label in sorted(groups)]
    return [p for round_ in zip(*picked) for p in round_] + [
        p for g in picked for p in g[min(map(len, picked)):]
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", type=Path)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--part", default="synthetic_plate", choices=sorted(SPECS))
    ap.add_argument("--vlm-per-label", type=int, default=10)
    ap.add_argument("--methods", default="trained_classifier,naive_vlm,inspection_cell")
    ap.add_argument("--mock", action="store_true", help="use MockVLM in the cell and skip naive_vlm (dry run)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    out = Path("results") / args.name
    out.mkdir(parents=True, exist_ok=True)
    info_path = out / "info.json"
    if args.report_only:
        print(write_report(out, json.loads(info_path.read_text())))
        return 0

    cfg = replace(CellConfig(), **PART_QUALITY[args.part])
    train, test = list_images(args.train), list_images(args.test)
    info = {
        "name": args.name, "train": str(args.train), "test": str(args.test),
        "n_train": len(train), "n_test": len(test), "model": cfg.model, "mock": args.mock, "part": args.part,
    }
    info_path.write_text(json.dumps(info, indent=2))

    preds = out / "predictions.jsonl"
    done = set()
    if preds.exists():
        done = {(r["method"], r["image"]) for r in map(json.loads, preds.read_text().splitlines())}

    methods = args.methods.split(",")
    if args.mock and "naive_vlm" in methods:
        methods.remove("naive_vlm")
    vlm_test = stratified(test, args.vlm_per_label)

    plan = []
    if "trained_classifier" in methods:
        plan.append((TrainedClassifier().fit(train), test))
    if "naive_vlm" in methods:
        plan.append((NaiveVLM(cfg), vlm_test))
    if "inspection_cell" in methods:
        # The presence check is calibrated on the training set only, then frozen.
        spec = SPECS[args.part]
        presence = calibrate(train, spec, label_of)
        presence.save(out / "presence.json")
        info["presence_train_accuracy"] = presence.train_accuracy
        info_path.write_text(json.dumps(info, indent=2))
        plan.append((InspectionCell(cfg, out / "runtime", presence, spec, mock=args.mock), vlm_test))

    for method, images in plan:
        todo = [p for p in images if (method.name, str(p)) not in done]
        print(f"\n{method.name}: {len(images) - len(todo)} done, {len(todo)} to go")
        for i, path in enumerate(todo, 1):
            label = label_of(path)
            row = {"method": method.name, "image": str(path), "label": label,
                   "expected": expected_decision(label).value, **timed(method, path)}
            with preds.open("a") as f:
                f.write(json.dumps(row) + "\n")
            mark = "ok " if row["decision"] == row["expected"] else "XX "
            print(f"  [{i}/{len(todo)}] {mark}{path.name:24s} {row['decision']:7s} {row['latency_s']:6.1f}s  {row['reason'][:70]}", flush=True)

    print("\nreport:", write_report(out, info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
