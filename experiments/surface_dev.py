"""Check a part's surface prompt on TRAINING photos only (never the test set), at board level:
good boards must have every tile CLEAN, crack/scratch boards at least one DAMAGED tile.

    python -m experiments.surface_dev data/real/train --part cardboard_board
"""

import argparse
from pathlib import Path

from cell.config import CellConfig
from cell.spec import SPECS, locate
from cell.validator import ValidationError, parse_region_answer
from cell.vlm import OllamaVLM

from .dataset import label_of, list_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train", type=Path)
    ap.add_argument("--part", default="cardboard_board", choices=sorted(SPECS))
    args = ap.parse_args()
    vlm = OllamaVLM(CellConfig())
    right = total = 0
    for path in list_images(args.train):
        label = label_of(path)
        if label not in ("good", "crack", "scratch"):
            continue
        answers = []
        for region in [r for r in locate(path, SPECS[args.part]) if r.kind == "surface"]:
            raw = vlm.inspect(path, region)
            try:
                result = parse_region_answer(raw, region)
                answers.append(f"{region.name.split()[1]}={result.answer}")
                print(f"    {path.name} {region.name}: {result.answer}  {result.description[:90]}", flush=True)
                if result.is_defect and label != "good":
                    break  # board already rejected
            except ValidationError as e:
                answers.append(f"{region.name.split()[1]}=INVALID")
                print(f"    {path.name} {region.name}: INVALID {e}", flush=True)
        damaged = any(a.endswith("DAMAGED") for a in answers)
        ok = damaged == (label != "good")
        right += ok
        total += 1
        print(f"{'ok' if ok else 'XX'} {path.name:18s} {' '.join(answers)}", flush=True)
    print(f"\nboards right: {right}/{total}")


if __name__ == "__main__":
    main()
