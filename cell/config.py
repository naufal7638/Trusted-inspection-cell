"""Every threshold the cell uses, in one place, so an engineer can audit and tune them."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CellConfig:
    # Local model served by Ollama. Nothing leaves this machine.
    model: str = "gemma3:4b"
    ollama_url: str = "http://localhost:11434"
    # Per region. CPU-only inference on a 4B vision model takes ~15-20 s per close-up.
    vlm_timeout_s: float = 120.0
    # Our prompts are ~400 tokens. Ollama's default context reserves far more memory
    # (the model grew to 7.3 GB and was OOM-killed on a 14 GB laptop).
    num_ctx: int = 2048
    # Ollama's memory grew ~100 MB per image until it was OOM-killed again (3.7 -> 8.4 GB).
    # Unloading the model every N calls resets it, for a ~5 s reload.
    unload_every_n_calls: int = 20

    # Image quality gate, checked before the AI ever sees the image.
    min_brightness: float = 40.0
    max_brightness: float = 225.0
    min_sharpness: float = 30.0

    # Circuit breaker: after this many AI failures in a row the cell runs in
    # degraded mode (every part goes to human review) and only probes the
    # model again every `probe_every_n_parts` parts.
    max_consecutive_vlm_failures: int = 3
    probe_every_n_parts: int = 5

    # Calibrated screw-presence check (cell/presence.py), fitted on the training set.
    presence_model_path: str = "models/presence.json"

    runtime_dir: str = "runtime"


# Image-quality limits depend on the camera, so each part spec has its own. Cardboard values
# are midpoints between the one dark / one blurry TRAINING photo and the dimmest / least sharp
# normal training photo (dark 134 vs 150; blurry 137 vs 322). One example each: a weak calibration.
PART_QUALITY = {
    "synthetic_plate": {},
    "cardboard_board": {"min_brightness": 142.0, "min_sharpness": 230.0},
}
