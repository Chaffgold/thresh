"""PyTorch training scripts for the flight-data training pipeline.

Two tracks: a point-cloud-to-3D-box detector (Track A) and an IMM
mode-probability classifier (Track B). Track B trains on filter
outputs, not raw ADS-B states (see `design.md` Decision 4).
"""
