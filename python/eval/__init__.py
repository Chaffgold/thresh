"""Evaluation harness for trained checkpoints.

Drives the full thresh tracker (via `thresh-py`) on held-out trajectory
splits and reports MOTA / MOTP / IDF1. ONNX-Rust parity smoke tests live
here too.
"""
