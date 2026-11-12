"""Run the inspection cell.

    python run.py --source samples --mock        # dry run, no model needed
    python run.py --source samples               # real run with local Gemma via Ollama
    python run.py --webcam --parts 10            # live camera, one frame every 3 s

E-stop: create the file runtime/ESTOP (e.g. `touch runtime/ESTOP`) and the line halts
before the next part and before any actuation. Delete it to clear.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import py_trees

from cell import audit
from cell.config import PART_QUALITY, CellConfig
from cell.presence import PresenceModel
from cell.spec import SPECS
from cell.sources import FolderSource, WebcamSource
from cell.tree import Cell, build_tree
from cell.vlm import MockVLM, OllamaVLM


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", type=Path, help="folder of part images")
    src.add_argument("--webcam", action="store_true", help="capture parts from the webcam")
    ap.add_argument("--parts", type=int, help="stop after this many webcam parts")
    ap.add_argument("--mock", action="store_true", help="use the filename-driven mock model")
    ap.add_argument("--model", default=CellConfig.model, help="Ollama model name")
    ap.add_argument("--part", default="synthetic_plate", choices=sorted(SPECS), help="which part spec to inspect against")
    ap.add_argument("--presence", help="presence model JSON (default: models/presence.json)")
    ap.add_argument("--show-tree", action="store_true", help="print the tree with node status after each part")
    args = ap.parse_args()

    cfg = replace(CellConfig(), model=args.model, **PART_QUALITY[args.part])
    runtime = Path(cfg.runtime_dir)
    source = (
        WebcamSource(runtime / "captures", max_parts=args.parts) if args.webcam else FolderSource(args.source)
    )
    vlm = MockVLM() if args.mock else OllamaVLM(cfg)
    presence_path = Path(args.presence or cfg.presence_model_path)
    presence = PresenceModel.load(presence_path) if presence_path.exists() else None
    if presence is None:
        print(f"warning: no presence model at {presence_path}; screws are checked by the AI alone")
    cell = Cell(cfg, source, vlm, spec=SPECS[args.part], presence=presence)
    tree = build_tree(cell)

    print(py_trees.display.unicode_tree(tree.root))
    print(f"model: {vlm.name}   run: {cell.run_id}\n")

    counts: dict[str, int] = {}
    while cell.stopped is None:
        cell.part = None
        tree.tick()
        part = cell.part
        if part is not None:
            entry = audit.record_part(runtime, part, vlm.name)
            counts[entry["decision"]] = counts.get(entry["decision"], 0) + 1
            latency = f"{part.vlm_latency_s:5.1f}s" if part.vlm_latency_s is not None else "   -  "
            print(f"{part.part_id}  {part.image_path.name:28s} {entry['decision']:7s} {latency}  {part.reason}")
        if args.show_tree:
            print(py_trees.display.unicode_tree(tree.root, show_status=True))
        if cell.degraded and part is not None and part.decided_by == "fallback":
            print("  ! degraded mode: AI is failing, parts are going to human review")

    print(f"\nline stopped: {cell.stopped}")
    print("totals:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none")
    print(f"audit log: {audit.audit_path(runtime)}   review queue: python review.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
