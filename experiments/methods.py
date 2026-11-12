"""The three methods compared in the experiment. Each one maps an image to PASS / REJECT / REVIEW.

  trained_classifier   baseline 1: logistic regression trained on the training set (classic ML)
  naive_vlm            baseline 2: Gemma looks at the whole part, reports defect + confidence
  inspection_cell      ours: the behaviour tree with spec regions, validator and rules
"""

import base64
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cell.config import CellConfig
from cell.presence import PresenceModel
from cell.tree import Cell, build_tree
from cell.validator import Decision
from cell.vlm import MockVLM, OllamaVLM

from .dataset import expected_decision, label_of

FEATURE_PX = 64


def _features(path: Path) -> np.ndarray:
    g = np.asarray(Image.open(path).convert("L").resize((FEATURE_PX, FEATURE_PX)), dtype=np.float64).ravel()
    return (g - g.mean()) / (g.std() + 1e-6)  # remove overall brightness so it learns shape, not exposure


class TrainedClassifier:
    name = "trained_classifier"

    def fit(self, train: list[Path]) -> "TrainedClassifier":
        X = np.stack([_features(p) for p in train])
        y = [label_of(p) for p in train]
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
        self.model.fit(X, y)
        return self

    def predict(self, path: Path) -> dict:
        label = self.model.predict(_features(path)[None])[0]
        return {"decision": expected_decision(label).value, "reason": f"classifier predicted {label}"}


NAIVE_PROMPT = """You are a quality inspector on a manufacturing line.
Inspect the part in the image for defects such as missing screws, loose or crooked
screws, cracks, scratches, or misaligned components.

Reply with JSON only:
{"defect": one of ["none", "missing_screw", "crack", "scratch", "misaligned", "other"],
  "confidence": a number from 0 to 1,
  "explanation": one short sentence an operator can read}
Use "none" if the part looks good. If you are unsure, lower the confidence."""

NAIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "defect": {"type": "string", "enum": ["none", "missing_screw", "crack", "scratch", "misaligned", "other"]},
        "confidence": {"type": "number"},
        "explanation": {"type": "string"},
    },
    "required": ["defect", "confidence", "explanation"],
}


class NaiveVLM:
    """Our first design: one whole-image question, trust the model's own confidence."""

    name = "naive_vlm"
    PASS_CONFIDENCE = 0.85
    REJECT_CONFIDENCE = 0.70

    def __init__(self, cfg: CellConfig):
        self.cfg = cfg

    def predict(self, path: Path) -> dict:
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": NAIVE_PROMPT, "images": [base64.b64encode(path.read_bytes()).decode()]}],
            "format": NAIVE_SCHEMA,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_ctx": self.cfg.num_ctx},
        }
        try:
            r = requests.post(f"{self.cfg.ollama_url}/api/chat", json=body, timeout=self.cfg.vlm_timeout_s)
            r.raise_for_status()
            answer = json.loads(r.json()["message"]["content"])
            defect, conf = answer["defect"], float(answer["confidence"])
        except Exception as e:  # any failure is a REVIEW, as in the real cell
            return {"decision": Decision.REVIEW.value, "reason": f"no usable answer: {e}"}
        if defect == "none":
            decision = Decision.PASS if conf >= self.PASS_CONFIDENCE else Decision.REVIEW
        else:
            decision = Decision.REJECT if conf >= self.REJECT_CONFIDENCE else Decision.REVIEW
        return {"decision": decision.value, "reason": f"{defect} @ {conf:.2f}: {answer.get('explanation', '')}"}


class _OneAtATime:
    def __init__(self):
        self.next: Path | None = None

    def next_image(self) -> Path | None:
        path, self.next = self.next, None
        return path


class InspectionCell:
    """The real system, unchanged: same tree, same validator, same rules."""

    name = "inspection_cell"

    def __init__(self, cfg: CellConfig, runtime: Path, presence: PresenceModel | None, spec, mock: bool = False):
        self.source = _OneAtATime()
        cfg = replace(cfg, runtime_dir=str(runtime))
        self.cell = Cell(cfg, self.source, MockVLM() if mock else OllamaVLM(cfg), spec=spec, presence=presence)
        self.tree = build_tree(self.cell)

    def predict(self, path: Path) -> dict:
        self.source.next = path
        self.cell.part = None
        self.tree.tick()
        part = self.cell.part
        return {
            "decision": part.decision.value,
            "reason": part.reason,
            "regions_checked": len(part.regions),
        }


def timed(method, path: Path) -> dict:
    t0 = time.monotonic()
    out = method.predict(path)
    out["latency_s"] = round(time.monotonic() - t0, 2)
    return out
