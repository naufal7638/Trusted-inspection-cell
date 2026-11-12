"""Vision-language model clients. They only return raw text; trusting it is the validator's job."""

import base64
from pathlib import Path

import requests

from .config import CellConfig
from .spec import DEFECT_ANSWER, OK_ANSWER, Region, crop

# "Describe first, then answer" matters. With a forced JSON schema, Gemma 3 4B gave the
# same answer for every crop; asked to describe first, it read each crop correctly.
PROMPTS = {
    # v2, chosen on training crops (experiments/prompt_dev.py): v1 called rotated screws
    # ("X" slots) empty; v2 reads screws correctly but still misses some empty holes,
    # which is why the deterministic presence check exists.
    "screw": f"""This is a close-up of one screw position on a metal plate.
An installed screw looks like a light round head with a cross-shaped slot. The cross can be turned to any angle, so it may look like a + or like an X.
An empty hole looks like a plain dark round hole, with no head and no slot.
First, describe in one sentence exactly what you see.
Then, on a new last line, write exactly one of:
ANSWER: {OK_ANSWER['screw']}   (a screw head is installed)
ANSWER: {DEFECT_ANSWER['screw']}   (the hole has no screw)""",
    "surface": f"""This is a close-up of a metal plate surface on a manufacturing line.
First, describe the surface in one sentence: is it smooth and uniform, or are there any lines, cracks, scratches or marks on it?
Then, on a new last line, write exactly one of:
ANSWER: {OK_ANSWER['surface']}
ANSWER: {DEFECT_ANSWER['surface']}""",
}


class VLMError(Exception):
    """The model could not produce an answer at all (timeout, server down, HTTP error)."""


class OllamaVLM:
    def __init__(self, cfg: CellConfig):
        self.cfg = cfg
        self.calls = 0

    @property
    def name(self) -> str:
        return f"ollama:{self.cfg.model}"

    def inspect(self, image_path: Path, region: Region) -> str:
        image_b64 = base64.b64encode(crop(image_path, region)).decode()
        self.calls += 1
        unload = self.calls % self.cfg.unload_every_n_calls == 0
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": region.prompt or PROMPTS[region.kind], "images": [image_b64]}],
            "stream": False,
            # Keep the model loaded between parts, except every N calls (see config).
            "keep_alive": 0 if unload else "30m",
            "options": {"temperature": 0, "num_ctx": self.cfg.num_ctx},
        }
        try:
            r = requests.post(
                f"{self.cfg.ollama_url}/api/chat", json=body, timeout=self.cfg.vlm_timeout_s
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except requests.Timeout:
            raise VLMError(f"model did not answer within {self.cfg.vlm_timeout_s:.0f}s")
        except requests.ConnectionError:
            raise VLMError("model server unreachable")
        except (requests.HTTPError, KeyError, ValueError) as e:
            raise VLMError(f"bad response from model server: {e}")


class MockVLM:
    """Stand-in model for tests and dry runs. It reads the answer from the file name,
    so it is NOT an inspector; it exists to exercise every path through the tree.

      missing_screw_*   -> top-right screw hole is empty
      crack_*           -> centre of the surface is damaged
      garbage_*         -> rambling answer with no ANSWER line
      timeout_*         -> raises VLMError
      anything else     -> every region OK
    """

    name = "mock"

    def inspect(self, image_path: Path, region: Region) -> str:
        stem = Path(image_path).stem
        if stem.startswith("timeout"):
            raise VLMError("model did not answer (simulated timeout)")
        if stem.startswith("garbage"):
            return "Sure! The part looks mostly fine I think."
        defect = (stem.startswith("missing_screw") and region.name == "screw top-right") or (
            stem.startswith("crack") and region.name == "surface center"
        )
        if defect:
            return f"I see a defect here.\nANSWER: {DEFECT_ANSWER[region.kind]}"
        return f"Looks normal.\nANSWER: {OK_ANSWER[region.kind]}"
