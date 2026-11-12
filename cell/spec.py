"""The part spec: what a good part looks like and where to look.

Positions are fractions of the image, so they assume the part sits in a fixed jig under a
fixed camera, as it would in a real inspection cell. The deterministic side decides WHERE
to look; the AI only answers a narrow question about each close-up.

Why close-ups: on a whole-part image, Gemma 3 4B said a plate with a missing screw was
fine at 95% confidence and missed a crack. On close-ups of each region it got every
screw right and flagged the crack.
"""

import io
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

# Each region kind has exactly one "OK" answer and one "defect" answer.
OK_ANSWER = {"screw": "SCREW", "surface": "CLEAN"}
DEFECT_ANSWER = {"screw": "EMPTY", "surface": "DAMAGED"}
DEFECT_NAME = {"screw": "missing screw", "surface": "surface damage"}


@dataclass(frozen=True)
class Region:
    name: str
    kind: str  # "screw" or "surface"
    box: tuple[float, float, float, float]  # left, top, right, bottom as fractions of width/height
    # Screw regions only: where to look for the screw (or hole). The box is re-centred on
    # what is found, so a part placed a little off the jig is still inspected in the right place.
    search: tuple[float, float, float, float] | None = None
    # The question the AI is asked about this region, when it differs from the default for
    # its kind. What counts as damage depends on the material, so it belongs to the part spec.
    prompt: str | None = None


def _box(left: int, top: int, right: int, bottom: int, size: int = 512) -> tuple[float, float, float, float]:
    return (left / size, top / size, right / size, bottom / size)


def _screw(name: str, x: int, y: int) -> Region:
    return Region(name, "screw", _box(x - 60, y - 60, x + 60, y + 60))


# Screws first: a missing screw is the most common defect and ends the check early.
PART_SPEC = (
    _screw("screw top-left", 120, 120),
    _screw("screw top-right", 392, 120),
    _screw("screw bottom-left", 120, 392),
    _screw("screw bottom-right", 392, 392),
    Region("surface top", "surface", _box(160, 70, 352, 190)),
    Region("surface left", "surface", _box(70, 160, 190, 352)),
    Region("surface center", "surface", _box(160, 160, 352, 352)),
    Region("surface right", "surface", _box(322, 160, 442, 352)),
    Region("surface bottom", "surface", _box(160, 322, 352, 442)),
)

# Real cardboard test board, photographed by a fixed 1920x1080 webcam looking straight down.
# Screw boxes are ~2.2 screw-head radii across; positions were measured on training photos.
def _real_screw(name: str, search: tuple[float, float, float, float]) -> Region:
    cx, cy = (search[0] + search[2]) / 2, (search[1] + search[3]) / 2
    hw, hh = 0.055, 0.055 * 1920 / 1080  # square in pixels
    return Region(name, "screw", (cx - hw, cy - hh, cx + hw, cy + hh), search)


# v2, chosen on training photos only. v1 (the metal-plate prompt) called normal cardboard
# texture "numerous small scratches" and rejected every good board.
CARDBOARD_SURFACE_PROMPT = f"""This is a close-up of a plain brown cardboard board on an inspection line.
Normal cardboard is NOT damage: its fibre texture, soft folds and creases, faint parallel ridges,
gentle shading, colour variation and small dark specks are all normal.
Damage is a distinct, sharp, dark line that stands out from the cardboard, such as a crack or a
pen or pencil scratch, straight or zig-zag.
First, describe in one sentence whether you see any distinct sharp dark line.
Then, on a new last line, write exactly one of:
ANSWER: {OK_ANSWER['surface']}
ANSWER: {DEFECT_ANSWER['surface']}"""


def _real_surface(name: str, box: tuple[float, float, float, float]) -> Region:
    return Region(name, "surface", box, prompt=CARDBOARD_SURFACE_PROMPT)


CARDBOARD_SPEC = (
    _real_screw("screw top-left", (0.26, 0.03, 0.42, 0.40)),
    _real_screw("screw top-right", (0.56, 0.03, 0.72, 0.40)),
    _real_screw("screw bottom-left", (0.26, 0.60, 0.42, 0.97)),
    _real_screw("screw bottom-right", (0.56, 0.60, 0.72, 0.97)),
    _real_surface("surface top", (0.37, 0.08, 0.57, 0.30)),
    _real_surface("surface left", (0.25, 0.32, 0.355, 0.66)),
    _real_surface("surface center", (0.37, 0.32, 0.57, 0.66)),
    _real_surface("surface right", (0.585, 0.32, 0.70, 0.66)),
    _real_surface("surface bottom", (0.37, 0.68, 0.57, 0.90)),
)

SPECS = {"synthetic_plate": PART_SPEC, "cardboard_board": CARDBOARD_SPEC}

LOCATE_WIDTH = 480


def locate(image_path: Path, spec: tuple[Region, ...]) -> tuple[Region, ...]:
    """Re-centre each screw region on the strongest local-contrast blob in its search window
    (a screw head or an empty hole both stand out from flat board). Deterministic."""
    if not any(r.search for r in spec):
        return spec
    img = Image.open(image_path).convert("L")
    w, h = LOCATE_WIDTH, round(LOCATE_WIDTH * img.height / img.width)
    g = np.asarray(img.resize((w, h)), dtype=np.float64)
    mean = uniform_filter(g, 9)
    std = np.sqrt(np.clip(uniform_filter(g * g, 9) - mean * mean, 0, None))
    contrast = uniform_filter(std, 25)  # about one screw-head across, so the peak lands on its centre
    out = []
    for r in spec:
        if r.search is None:
            out.append(r)
            continue
        x0, y0, x1, y1 = (round(r.search[0] * w), round(r.search[1] * h), round(r.search[2] * w), round(r.search[3] * h))
        yy, xx = np.unravel_index(contrast[y0:y1, x0:x1].argmax(), (y1 - y0, x1 - x0))
        cx, cy = (x0 + xx) / w, (y0 + yy) / h
        hw, hh = (r.box[2] - r.box[0]) / 2, (r.box[3] - r.box[1]) / 2
        out.append(replace(r, box=(cx - hw, cy - hh, cx + hw, cy + hh)))
    return tuple(out)


SCREW_CROP_PX = 360  # screw crops are upscaled; the model reads small features better


def crop(image_path: Path, region: Region) -> bytes:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    left, top, right, bottom = region.box
    out = img.crop((round(left * w), round(top * h), round(right * w), round(bottom * h)))
    if region.kind == "screw":
        out = out.resize((SCREW_CROP_PX, SCREW_CROP_PX))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()
