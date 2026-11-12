"""Deterministic checks between the AI and the line. No model output reaches an actuator
without passing through here."""

import re
from dataclasses import dataclass
from enum import Enum

from .spec import DEFECT_ANSWER, DEFECT_NAME, OK_ANSWER, Region

MAX_DESCRIPTION_CHARS = 300
ANSWER_LINE = re.compile(r"^\W*ANSWER\W*\s*([A-Za-z]+)")


class Decision(str, Enum):
    PASS = "PASS"
    REJECT = "REJECT"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class RegionResult:
    region: str
    kind: str
    answer: str
    description: str  # what the model said it saw; shown to the operator

    @property
    def is_defect(self) -> bool:
        return self.answer == DEFECT_ANSWER[self.kind]


class ValidationError(ValueError):
    """The model answered, but the answer is not something we can act on."""


def parse_region_answer(raw: str, region: Region) -> RegionResult:
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        raise ValidationError(f"AI gave an empty answer for {region.name}")
    m = ANSWER_LINE.match(lines[-1])
    if not m:
        raise ValidationError(f"AI answer for {region.name} has no ANSWER line")
    answer = m.group(1).upper()
    allowed = (OK_ANSWER[region.kind], DEFECT_ANSWER[region.kind])
    if answer not in allowed:
        raise ValidationError(f"AI answered {answer!r} for {region.name}, expected one of {allowed}")
    # The description is required: it is the evidence an operator reads, and asking the
    # model to look before it answers is what makes the answer usable at all.
    description = " ".join(lines[:-1])[:MAX_DESCRIPTION_CHARS]
    if not description:
        raise ValidationError(f"AI gave no description for {region.name}")
    return RegionResult(region.name, region.kind, answer, description)


def decide(
    results: list[RegionResult],
    spec: tuple[Region, ...],
    presence: dict[str, tuple[str, float]] | None = None,
) -> tuple[Decision, str]:
    """Fixed rules that turn validated region answers into a line action.

    One confirmed defect is enough to reject: a false reject costs a re-check.
    Passing needs every region in the spec checked and OK: a false pass ships a defect.

    `presence` is the deterministic screw check (region -> (SCREW|EMPTY|UNSURE, p_screw)).
    When it is given, a screw is only OK if BOTH it and the AI say so:
        check EMPTY                -> REJECT (no AI needed)
        check UNSURE + AI EMPTY    -> REJECT
        check SCREW  + AI EMPTY    -> REVIEW (they disagree)
        check UNSURE + AI SCREW    -> REVIEW (can't confirm)
    """
    presence = presence or {}
    ai = {r.region: r for r in results}
    rejects, reviews = [], []

    for name, (state, p) in presence.items():
        if state == "EMPTY":
            rejects.append(f"missing screw at {name}: presence check found an empty hole (p_screw {p:.2f})")
    for r in results:
        if not r.is_defect:
            continue
        state = presence.get(r.region, (None, None))[0] if r.kind == "screw" else None
        if state == "SCREW":
            reviews.append(f"checks disagree at {r.region}: AI saw \"{r.description}\" but presence check found a screw")
        else:
            rejects.append(f"{DEFECT_NAME[r.kind]} at {r.region}: AI saw \"{r.description}\"")
    for name, (state, p) in presence.items():
        if state == "UNSURE" and name in ai and not ai[name].is_defect:
            reviews.append(f"presence check unsure at {name} (p_screw {p:.2f}); AI says screw")

    if rejects:
        return Decision.REJECT, rejects[0]
    if reviews:
        return Decision.REVIEW, reviews[0]
    checked = set(ai)
    unchecked = [region.name for region in spec if region.name not in checked]
    if unchecked:
        return Decision.REVIEW, f"not every region was checked ({', '.join(unchecked)})"
    return Decision.PASS, f"all {len(spec)} regions checked and OK" + (" (screws confirmed by both checks)" if presence else "")
