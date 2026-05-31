"""Tests for the Track A class taxonomy (design.md Decision 8). Torch-free."""

from __future__ import annotations

from training.classes import NUM_CLASSES, TargetClass, class_for_adsb_category


def test_num_classes_matches_taxonomy() -> None:
    assert NUM_CLASSES == 5
    # Index alignment with the Rust synth / ONNX contract.
    assert TargetClass.LIGHT_FIXED_WING == 0
    assert TargetClass.OTHER == 4


def test_adsb_category_mapping() -> None:
    assert class_for_adsb_category("A1") == TargetClass.LIGHT_FIXED_WING
    assert class_for_adsb_category("A2") == TargetClass.LIGHT_FIXED_WING
    assert class_for_adsb_category("A3") == TargetClass.HEAVY_FIXED_WING
    assert class_for_adsb_category("A5") == TargetClass.HEAVY_FIXED_WING
    assert class_for_adsb_category("A7") == TargetClass.ROTORCRAFT
    assert class_for_adsb_category("B2") == TargetClass.GLIDER_BALLOON_UAV


def test_case_insensitive_and_whitespace() -> None:
    assert class_for_adsb_category(" a3 ") == TargetClass.HEAVY_FIXED_WING


def test_unknown_and_none_map_to_other() -> None:
    assert class_for_adsb_category(None) == TargetClass.OTHER
    assert class_for_adsb_category("Z9") == TargetClass.OTHER
    assert class_for_adsb_category("") == TargetClass.OTHER
    # Military / blocked codes fall through to OTHER.
    assert class_for_adsb_category("C3") == TargetClass.OTHER
