"""Compare screw prompts on TRAINING-set crops only (never the test set).

    python -m experiments.prompt_dev data/synthetic/train --per-class 6
"""

import argparse
import base64
from dataclasses import replace
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from cell.config import CellConfig
from cell.spec import PART_SPEC, crop
from cell.validator import ValidationError, parse_region_answer
from cell.vlm import PROMPTS

from .dataset import label_of, list_images

SCREWS = [r for r in PART_SPEC if r.kind == "screw"]


def is_empty(image_path: Path, region) -> bool:
    """Ground truth for synthetic crops: an empty hole has a dark centre, a screw head a light one."""
    img = Image.open(image_path).convert("L")
    w, h = img.size
    l, t, r, b = region.box
    cx, cy = (l + r) / 2 * w, (t + b) / 2 * h
    centre = np.asarray(img.crop((cx - 6, cy - 6, cx + 6, cy + 6)), dtype=float)
    return centre.mean() < 80


def dev_crops(folder: Path, per_class: int) -> list[tuple[Path, object, str]]:
    screws, empties = [], []
    for p in list_images(folder):
        if label_of(p) not in ("good", "missing_screw"):
            continue
        for region in SCREWS:
            (empties if is_empty(p, region) else screws).append((p, region))
    pick = lambda xs, n: [xs[i] for i in np.linspace(0, len(xs) - 1, n).astype(int)]
    return [(p, r, "SCREW") for p, r in pick(screws, per_class)] + [(p, r, "EMPTY") for p, r in pick(empties, per_class)]


def ask(cfg: CellConfig, prompt: str, image_path: Path, region) -> str:
    body = {
        "model": cfg.model, "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": cfg.num_ctx},
        "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(crop(image_path, region)).decode()]}],
    }
    return requests.post(f"{cfg.ollama_url}/api/chat", json=body, timeout=cfg.vlm_timeout_s).json()["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train", type=Path)
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--prompt-file", type=Path, help="candidate prompt to compare against the current one")
    args = ap.parse_args()
    cfg = CellConfig()
    prompts = {"current": PROMPTS["screw"]}
    if args.prompt_file:
        prompts["candidate"] = args.prompt_file.read_text()
    crops = dev_crops(args.train, args.per_class)
    for name, prompt in prompts.items():
        right = 0
        for p, region, truth in crops:
            try:
                got = parse_region_answer(ask(cfg, prompt, p, region), region).answer
            except ValidationError:
                got = "INVALID"
            right += got == truth
            print(f"  {name:9s} {p.name:22s} {region.name:18s} truth={truth:5s} got={got}", flush=True)
        print(f"{name}: {right}/{len(crops)} correct\n", flush=True)


if __name__ == "__main__":
    main()
