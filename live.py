"""Live webcam station: place a board, press SPACE, the cell inspects it.

    .venv/bin/python live.py                    # real Gemma via Ollama
    .venv/bin/python live.py --mock             # no model, to check the camera and framing
    .venv/bin/python live.py --device 2         # pick a camera (default: the first full-HD one)

The preview shows where the part spec expects the board: put each screw inside a yellow box.
The top bar re-checks the live view every second without the AI: image sharpness, and each screw
(green box = SCREW, red = EMPTY, orange = UNSURE). Adjust focus/light until a good board is all green.
Pink crosses mark where the screws sat in the training photos; the second line says how to move the
camera or board until it reads "framing OK" (the surface tiles only cover the board then).
SPACE captures and inspects (the preview keeps running), Q quits. Results go to the same audit
log as run.py, so the dashboard shows each part with its boxes and the AI's words.
The e-stop works as usual: the dashboard button halts the next inspection.

Station calibration (when the camera or light differs from the training photos): press G on a
good board, and 1/2/3/4 on a board whose top-left/top-right/bottom-left/bottom-right screw is
missing (each corner at least once), ~5 good and ~6 missing in total, then
    .venv/bin/python live.py --calibrate
fits the screw check on those photos only (models/presence_live.json; the training one is kept).
"""

import argparse
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2

from cell import audit
from cell.config import PART_QUALITY, CellConfig
from cell.presence import PresenceModel, calibrate
from cell.quality import measure
from cell.spec import SPECS, locate
from experiments.dataset import label_of, list_images
from cell.tree import Cell, build_tree
from cell.vlm import MockVLM, OllamaVLM

COLOUR = {"PASS": (74, 163, 22), "REJECT": (38, 38, 220), "REVIEW": (0, 165, 255)}  # BGR
YELLOW, GREY, WHITE = (0, 220, 255), (170, 170, 170), (255, 255, 255)
PREVIEW_WIDTH = 1280
# Station calibration: the screw check learns what a screw looks like under THIS camera and light.
CALIB_DIR = Path("data/live_calib")
LIVE_PRESENCE = Path("models/presence_live.json")
# G = good board; 1-4 = one screw missing at that corner (spec order: TL, TR, BL, BR)
CORNERS = ("TL", "TR", "BL", "BR")
CALIB_KEYS = {ord("g"): "good", **{ord(str(i + 1)): f"missing_screw_{c}" for i, c in enumerate(CORNERS)}}


def empty_corner(path: Path) -> int | None:
    """missing_screw_TR_03.jpg -> 1 (the operator said which corner was empty)."""
    parts = Path(path).stem.split("_")
    return CORNERS.index(parts[2]) if len(parts) > 2 and parts[2] in CORNERS else None
# Gemma runs on the same CPU. Keep OpenCV to one thread and slow the preview while inspecting,
# or the model's threads wait on ours (a crop went from ~20 s to over the 120 s timeout).
cv2.setNumThreads(1)


class OneShot:
    """Source that hands the tree exactly one captured image per tick."""

    def __init__(self):
        self.next: Path | None = None

    def next_image(self) -> Path | None:
        path, self.next = self.next, None
        return path


class Station:
    def __init__(self, cell: Cell, source: OneShot, model_name: str):
        self.cell, self.source, self.model_name = cell, source, model_name
        self.tree = build_tree(cell)
        self.busy_since: float | None = None
        self.result = "Place the board, screws inside the yellow boxes, then press SPACE"
        self.colour = WHITE

    def inspect(self, path: Path) -> None:
        self.busy_since = time.monotonic()
        threading.Thread(target=self._run, args=(path,), daemon=True).start()

    def _run(self, path: Path) -> None:
        try:
            self.source.next = path
            self.cell.part = None
            self.cell.stopped = None  # each SPACE is a fresh request; the e-stop is checked again
            self.tree.tick()
            part = self.cell.part
            if part is None:
                self.result, self.colour = f"LINE STOPPED: {self.cell.stopped}", COLOUR["REJECT"]
            else:
                entry = audit.record_part(self.cell.runtime, part, self.model_name)
                took = time.monotonic() - self.busy_since
                self.result = f"{entry['decision']} ({took:.0f}s): {part.reason}"
                self.colour = COLOUR[entry["decision"]]
                print(f"{part.part_id}  {path.name}  {self.result}", flush=True)
        except Exception as e:  # keep the camera running; show what went wrong
            self.result, self.colour = f"error: {e}", COLOUR["REJECT"]
        finally:
            self.busy_since = None


def training_layout(spec, train_dir: Path):
    """Average screw centres on the good TRAINING photos: where the camera framing must put them.
    The surface tiles are fixed, so they only cover the board when the framing matches."""
    goods = [p for p in list_images(train_dir) if label_of(p) == "good"] if train_dir.exists() else []
    if not goods:
        return None
    pts = [[((r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2) for r in locate(p, spec) if r.kind == "screw"]
           for p in goods]
    return [tuple(sum(c) / len(c) for c in zip(*corner)) for corner in zip(*pts)]


def framing_advice(found, target) -> str:
    """Compare the found screw centres (TL, TR, BL, BR) with the training layout."""
    size = lambda pts: ((pts[1][0] - pts[0][0] + pts[3][0] - pts[2][0]) / 2, (pts[2][1] - pts[0][1] + pts[3][1] - pts[1][1]) / 2)
    centre = lambda pts: (sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4)
    (fw, fh), (tw, th) = size(found), size(target)
    scale = (fw / tw + fh / th) / 2
    (fx, fy), (tx, ty) = centre(found), centre(target)
    tips = []
    if scale < 0.95:
        tips.append(f"board too small ({scale:.0%}): camera LOWER")
    elif scale > 1.05:
        tips.append(f"board too big ({scale:.0%}): camera HIGHER")
    if abs(fx - tx) > 0.025:
        tips.append("move board LEFT" if fx > tx else "move board RIGHT")
    if abs(fy - ty) > 0.025:
        tips.append("move board UP" if fy > ty else "move board DOWN")
    return "framing OK" if not tips else ", ".join(tips)


class SetupCheck:
    """Once a second, run the cheap deterministic checks on the live view (no AI), so the operator
    can adjust focus, light and placement until a good board reads sharp and every screw reads SCREW."""

    STATE_COLOUR = {"SCREW": (74, 163, 22), "EMPTY": (38, 38, 220), "UNSURE": (0, 165, 255)}

    def __init__(self, cell: Cell, path: Path, target=None):
        self.cell, self.path, self.target = cell, path, target
        self.text, self.framing, self.screws, self._busy, self._last = "", "", [], False, 0.0

    def maybe_run(self, frame) -> None:
        if self._busy or time.monotonic() - self._last < 1.0:
            return
        self._busy, self._last = True, time.monotonic()
        cv2.imwrite(str(self.path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            cfg = self.cell.cfg
            q = measure(self.path)
            regions = [r for r in locate(self.path, self.cell.spec) if r.kind == "screw"]
            self.screws = [(r, self.cell.presence.check(self.path, r)[0]) for r in regions]
            if self.target:
                found = [((r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2) for r in regions]
                self.framing = framing_advice(found, self.target)
            ok = "ok" if q.sharpness >= cfg.min_sharpness else "LOW"
            self.text = (f"setup: sharpness {q.sharpness:.0f} (need {cfg.min_sharpness:.0f}) {ok} | "
                         f"brightness {q.brightness:.0f} | screws: " + " ".join(s[:2] for _, s in self.screws))
        finally:
            self._busy = False

    def draw(self, frame) -> None:
        h, w = frame.shape[:2]
        for r, state in self.screws:
            b = r.box
            cv2.rectangle(frame, (round(b[0] * w), round(b[1] * h)), (round(b[2] * w), round(b[3] * h)),
                          self.STATE_COLOUR[state], 3)
        for x, y in self.target or []:  # where the screws sat in the training photos
            cv2.drawMarker(frame, (round(x * w), round(y * h)), (255, 0, 255), cv2.MARKER_CROSS, 60, 3)
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.putText(frame, self.text, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
        ok = self.framing == "framing OK"
        cv2.putText(frame, self.framing, (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (74, 163, 22) if ok else (0, 165, 255), 2, cv2.LINE_AA)


def draw_guide(frame, spec) -> None:
    h, w = frame.shape[:2]
    for r in spec:
        box = r.search or r.box
        p0, p1 = (round(box[0] * w), round(box[1] * h)), (round(box[2] * w), round(box[3] * h))
        cv2.rectangle(frame, p0, p1, YELLOW if r.kind == "screw" else GREY, 3 if r.kind == "screw" else 1)


def banner(frame, text: str, colour) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, text[:110], (16, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)


def open_camera(device: int):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # needed for full HD on most webcams
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


def find_camera() -> int:
    """The cardboard spec was calibrated on a 1920x1080 USB webcam; laptop cameras are usually 720p."""
    for device in range(6):
        cap = open_camera(device)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok and frame.shape[1] == 1920:
            return device
    return 0


def save_calibration_photo(frame, label: str) -> str:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    n = 1 + len(list(CALIB_DIR.glob(f"{label}_*.jpg")))
    path = CALIB_DIR / f"{label}_{n:02d}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    labels = [label_of(p) for p in CALIB_DIR.glob("*.jpg")]
    return (f"saved {path.name}   calibration photos: {labels.count('good')} good, "
            f"{labels.count('missing_screw')} missing")


def run_calibration(spec) -> int:
    """Fit the screw check on the station's own calibration photos, exactly as experiment.py
    fits it on the training set, and save it next to the original model (which is kept)."""
    images = list_images(CALIB_DIR) if CALIB_DIR.exists() else []
    labels = [label_of(p) for p in images]
    if labels.count("good") < 3 or labels.count("missing_screw") < 3:
        print(f"need at least 3 good and 3 missing-screw photos in {CALIB_DIR}, "
              f"have {labels.count('good')} good, {labels.count('missing_screw')} missing")
        return 1
    model = calibrate(images, spec, label_of, empty_corner)
    model.save(LIVE_PRESENCE)
    print(f"fitted on {model.n_train_crops} screw positions from {len(images)} photos, "
          f"accuracy on them {model.train_accuracy:.0%}; saved {LIVE_PRESENCE}")
    print("each photo checked by a model fitted WITHOUT it (an honest preview of live behaviour):")
    wrong = 0
    for p in images:
        held_out = calibrate([q for q in images if q != p], spec, label_of, empty_corner)
        states = [held_out.check(p, r)[0] for r in locate(p, spec) if r.kind == "screw"]
        truth = ["EMPTY" if i == empty_corner(p) else "SCREW" for i in range(4)]
        wrong += sum(s != t for s, t in zip(states, truth))
        print(f"  {p.name:26s} TL/TR/BL/BR: {' '.join(states)}")
    print(f"screw positions not read correctly: {wrong}/{4 * len(images)} (UNSURE counts as not correct; it goes to a human)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, help="camera number (default: the first full-HD camera)")
    ap.add_argument("--part", default="cardboard_board", choices=sorted(SPECS))
    ap.add_argument("--presence", help=f"screw check model (default: {LIVE_PRESENCE} if calibrated, else the training one)")
    ap.add_argument("--mock", action="store_true", help="use the stand-in model (camera and framing check only)")
    ap.add_argument("--calibrate", action="store_true", help=f"fit the screw check on the G/M photos in {CALIB_DIR}")
    args = ap.parse_args()

    cfg = replace(CellConfig(), **PART_QUALITY[args.part])
    spec = SPECS[args.part]
    if args.calibrate:
        return run_calibration(spec)
    if args.presence is None:
        args.presence = str(LIVE_PRESENCE if LIVE_PRESENCE.exists() else "models/presence_cardboard.json")
    print(f"screw check: {args.presence}")
    vlm = MockVLM() if args.mock else OllamaVLM(cfg)
    source = OneShot()
    cell = Cell(cfg, source, vlm, spec=spec, presence=PresenceModel.load(args.presence))
    station = Station(cell, source, vlm.name)
    captures = Path(cfg.runtime_dir) / "captures"
    captures.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        args.device = find_camera()
    setup = SetupCheck(cell, captures / "setup_check.jpg", training_layout(spec, Path("data/real/train")))
    cap = open_camera(args.device)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        raise SystemExit(f"camera {args.device} gave no image; try another --device")
    print(f"camera {args.device}: {frame.shape[1]}x{frame.shape[0]}   model: {vlm.name}   SPACE inspect, Q quit")
    if frame.shape[1] != 1920:
        print("  note: the cardboard spec was calibrated on 1920x1080 photos; other resolutions are untested")

    title = "Inspection station (SPACE inspect, G / 1-4 calibration photo, Q quit)"
    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera stopped giving images")
            break
        view = frame.copy()
        draw_guide(view, spec)
        if station.busy_since is None:
            setup.maybe_run(frame)
            setup.draw(view)
        if station.busy_since is not None:
            banner(view, f"INSPECTING... {time.monotonic() - station.busy_since:.0f}s", YELLOW)
        else:
            banner(view, station.result, station.colour)
        cv2.imshow(title, cv2.resize(view, (PREVIEW_WIDTH, round(PREVIEW_WIDTH * view.shape[0] / view.shape[1]))))

        key = cv2.waitKey(30 if station.busy_since is None else 250) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" ") and station.busy_since is None:
            path = captures / f"live_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            station.inspect(path)
        if key in CALIB_KEYS and station.busy_since is None:
            station.result, station.colour = save_calibration_photo(frame, CALIB_KEYS[key]), WHITE

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
