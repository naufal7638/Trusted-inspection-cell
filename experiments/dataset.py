"""Labelled datasets for the experiments.

Labels come from the file name prefix, so real photos work the same way:
    good_*, missing_screw_*, crack_*, scratch_*, misaligned_*   (part condition)
    dark_*, blurry_*                                             (unusable image)

The synthetic generator varies position, lighting, background, plate colour, screw
rotation, noise and defect placement, with different random seeds for train and test.
"""

import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from cell.validator import Decision

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
DEFECT_LABELS = ("missing_screw", "crack", "scratch", "misaligned")
BAD_IMAGE_LABELS = ("dark", "blurry")
SCREWS = [(120, 120), (392, 120), (120, 392), (392, 392)]


def label_of(path: Path) -> str:
    stem = Path(path).stem.lower()
    for label in ("good", *DEFECT_LABELS, *BAD_IMAGE_LABELS):
        if re.match(rf"{label}(_|\d|$)", stem):
            return label
    raise ValueError(f"cannot tell the label of {path.name}; name it like good_01.jpg")


def expected_decision(label: str) -> Decision:
    if label == "good":
        return Decision.PASS
    if label in BAD_IMAGE_LABELS:
        return Decision.REVIEW
    return Decision.REJECT


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def draw_part(rng: np.random.Generator, label: str) -> Image.Image:
    bg = int(rng.integers(160, 216))
    img = Image.new("RGB", (512, 512), (bg, bg, bg))
    d = ImageDraw.Draw(img)
    dx, dy = (int(v) for v in rng.integers(-8, 9, 2))
    shade = int(rng.integers(-15, 16))
    plate = (150 + shade, 155 + shade, 165 + shade)
    d.rounded_rectangle((60 + dx, 60 + dy, 452 + dx, 452 + dy), radius=24, fill=plate, outline=(70, 70, 80), width=4)

    missing = int(rng.integers(0, 4)) if label == "missing_screw" else None
    for i, (x, y) in enumerate(SCREWS):
        x, y = x + dx, y + dy
        if i == missing:
            r = int(rng.integers(11, 16))
            d.ellipse((x - r, y - r, x + r, y + r), fill=(30, 30, 30))
            continue
        d.ellipse((x - 26, y - 26, x + 26, y + 26), fill=(205, 205, 210), outline=(60, 60, 60), width=3)
        a = rng.uniform(0, math.pi / 2)
        for t in (a, a + math.pi / 2):
            ox, oy = 16 * math.cos(t), 16 * math.sin(t)
            d.line((x - ox, y - oy, x + ox, y + oy), fill=(60, 60, 60), width=5)

    if label == "crack":
        x, y = rng.uniform(180, 330) + dx, rng.uniform(170, 260) + dy
        points = [(x, y)]
        for _ in range(int(rng.integers(4, 7))):
            x = float(np.clip(x + rng.uniform(-30, 30), 170, 340))
            y = float(np.clip(y + rng.uniform(20, 45), 170, 345))
            points.append((x, y))
        d.line(points, fill=(40, 40, 40), width=int(rng.integers(3, 5)))
    elif label == "scratch":
        x, y = rng.uniform(180, 300) + dx, rng.uniform(180, 320) + dy
        t = rng.uniform(0, math.pi)
        length = rng.uniform(60, 120)
        tone = tuple(max(0, c - 55) for c in plate)
        d.line((x, y, x + length * math.cos(t), y + length * math.sin(t)), fill=tone, width=2)

    arr = np.asarray(img, dtype=np.float64) * rng.uniform(0.85, 1.15)
    arr += rng.normal(0, 3, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.8)))


def make_synthetic(out: Path, per_label: int, seed: int, labels=("good", "missing_screw", "crack", "scratch")) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for label in labels:
        for i in range(1, per_label + 1):
            draw_part(rng, label).save(out / f"{label}_{i:03d}.png")
