"""Deterministic screw-presence check: a second opinion that does not depend on the AI.

Gemma 3 4B could not reliably tell a screw from an empty hole once screw rotation and
lighting varied (it described an empty hole as "a round-headed screw with a cross-shaped
slot"). Presence is a measurable question, so it gets a measurable check:

  centre_ratio   mean brightness of the centre disc / mean of the ring of plate around it
  centre_detail  mean edge strength in the centre / ring mean (slot edges, head rim)
  centre_colour  saturation of the centre minus the ring (grey metal vs. orange cardboard)

A logistic regression on these three numbers is fitted on the TRAINING set only and saved to
JSON; at run time it is a fixed formula with no randomness. Scores near 0.5 are UNSURE,
and an UNSURE screw can never be passed.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .spec import Region, locate

SIZE = 96  # crops are resampled so the features don't depend on camera resolution


def region_crop(image_path: Path, region: Region) -> tuple[np.ndarray, np.ndarray]:
    """Grey level and saturation (0..1) of the region, resampled to SIZE x SIZE."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    left, top, right, bottom = region.box
    out = img.crop((round(left * w), round(top * h), round(right * w), round(bottom * h))).resize((SIZE, SIZE))
    grey = np.asarray(out.convert("L"), dtype=np.float64)
    sat = np.asarray(out.convert("HSV"), dtype=np.float64)[..., 1] / 255
    return grey, sat


def features(image_path: Path, region: Region) -> np.ndarray:
    g, sat = region_crop(image_path, region)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    r = np.hypot(yy - (SIZE - 1) / 2, xx - (SIZE - 1) / 2) / SIZE
    ring = (r > 0.36) & (r < 0.46)
    ring_mean = g[ring].mean() + 1e-6
    gy, gx = np.gradient(g)
    edges = np.hypot(gy, gx)
    return np.array([
        g[r < 0.2].mean() / ring_mean,
        edges[r < 0.3].mean() / ring_mean,
        sat[r < 0.2].mean() - sat[ring].mean(),
    ])


@dataclass(frozen=True)
class PresenceModel:
    weights: tuple[float, ...]
    bias: float
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    unsure_band: float = 0.2  # p_screw within 0.5 +/- band is UNSURE
    train_accuracy: float | None = None
    n_train_crops: int | None = None

    def p_screw(self, image_path: Path, region: Region) -> float:
        z = (features(image_path, region) - np.array(self.mean)) / np.array(self.scale)
        return float(1 / (1 + np.exp(-(z @ np.array(self.weights) + self.bias))))

    def check(self, image_path: Path, region: Region) -> tuple[str, float]:
        p = self.p_screw(image_path, region)
        if p >= 0.5 + self.unsure_band:
            return "SCREW", p
        if p <= 0.5 - self.unsure_band:
            return "EMPTY", p
        return "UNSURE", p

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PresenceModel":
        d = json.loads(Path(path).read_text())
        return cls(**{**d, "weights": tuple(d["weights"]), "mean": tuple(d["mean"]), "scale": tuple(d["scale"])})


def calibrate(images: list[Path], spec: tuple[Region, ...], label_of, empty_index_of=None) -> PresenceModel:
    """Fit on training images. `good_*` photos have all screws; `missing_screw_*` photos
    have exactly one missing. `empty_index_of(path)` names which screw position (in spec order)
    is empty; without it, the empty one is taken to be the position that looks least like the
    screws in the good photos (largest standardised distance from their average)."""
    from sklearn.linear_model import LogisticRegression

    per_image = []
    for path in images:
        label = label_of(path)
        if label in ("good", "missing_screw"):
            screws = [r for r in locate(path, spec) if r.kind == "screw"]
            per_image.append((label, [features(path, r) for r in screws], path))
    good = np.array([f for label, feats, _ in per_image if label == "good" for f in feats])
    g_mean, g_std = good.mean(axis=0), good.std(axis=0) + 0.01

    X, y = [], []
    for label, feats, path in per_image:
        empty = -1
        if label == "missing_screw":
            known = empty_index_of(path) if empty_index_of else None
            empty = known if known is not None else int(np.argmax([(((f - g_mean) / g_std) ** 2).sum() for f in feats]))
        X += feats
        y += [0 if i == empty else 1 for i in range(len(feats))]
    X = np.array(X)
    mean, scale = X.mean(axis=0), X.std(axis=0) + 0.01
    clf = LogisticRegression(class_weight="balanced").fit((X - mean) / scale, y)
    return PresenceModel(
        weights=tuple(float(w) for w in clf.coef_[0]),
        bias=float(clf.intercept_[0]),
        mean=tuple(float(m) for m in mean),
        scale=tuple(float(s) for s in scale),
        train_accuracy=float(clf.score((X - mean) / scale, y)),
        n_train_crops=len(y),
    )
