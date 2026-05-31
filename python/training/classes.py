"""Track A detector class taxonomy (design.md Decision 8).

Maps ADS-B emitter categories to a 5-class thresh-internal taxonomy. The indices
match the Rust side: the synth `gt_classes` (crates/thresh-synth, CLASS_BOX_DIMS
/ CLASS_RCS_M2) and the ONNX detector's `classes` output tensor in
`[0, NUM_CLASSES)`. Torch-free so it runs in the default CI lane.
"""

from __future__ import annotations

from enum import IntEnum


class TargetClass(IntEnum):
    """Detector output classes. Index-aligned with the Rust synth/ONNX contract."""

    LIGHT_FIXED_WING = 0
    HEAVY_FIXED_WING = 1
    ROTORCRAFT = 2
    GLIDER_BALLOON_UAV = 3
    OTHER = 4


#: Number of classes (matches NUM_CLASSES in scripts/generate_test_model.py).
NUM_CLASSES = len(TargetClass)

#: ADS-B emitter category → TargetClass (design.md Decision 8).
_ADSB_CATEGORY_TO_CLASS: dict[str, TargetClass] = {
    "A1": TargetClass.LIGHT_FIXED_WING,
    "A2": TargetClass.LIGHT_FIXED_WING,
    "A3": TargetClass.HEAVY_FIXED_WING,
    "A4": TargetClass.HEAVY_FIXED_WING,
    "A5": TargetClass.HEAVY_FIXED_WING,
    "A7": TargetClass.ROTORCRAFT,
    "B1": TargetClass.GLIDER_BALLOON_UAV,
    "B2": TargetClass.GLIDER_BALLOON_UAV,
    "B6": TargetClass.GLIDER_BALLOON_UAV,
}


def class_for_adsb_category(category: str | None) -> TargetClass:
    """Map an ADS-B emitter category (e.g. ``"A3"``) to a [`TargetClass`].

    Unknown categories, military/blocked codes, and ``None`` map to
    [`TargetClass.OTHER`] per Decision 8.
    """
    if category is None:
        return TargetClass.OTHER
    return _ADSB_CATEGORY_TO_CLASS.get(category.strip().upper(), TargetClass.OTHER)
