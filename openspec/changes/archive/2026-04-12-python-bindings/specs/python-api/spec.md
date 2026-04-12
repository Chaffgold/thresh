## Capability: Python API Bindings

### Overview

The Python package exposes the multi-object tracker, filter types, and MOT evaluation metrics as a pip-installable library built with PyO3, enabling Python-based integration and experimentation.

## ADDED Requirements

### Requirement: PyO3 pip-installable Python package

The system MUST expose the multi-object tracker, filter types, and MOT evaluation metrics as a pip-installable Python package via PyO3.

#### Scenario: Python tracker step with numpy arrays

**WHEN** `thresh.MultiObjectTracker` is instantiated in Python and `step()` is called with a numpy detection array

**THEN** the tracker processes the detections through the full tracking pipeline

**SHALL** return confirmed track positions as a numpy array
