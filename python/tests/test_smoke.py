"""Phase 1 smoke test: verify the four top-level packages import."""

import acquisition
import eval as eval_pkg
import export
import training


def test_packages_importable() -> None:
    assert acquisition.__doc__ is not None
    assert training.__doc__ is not None
    assert export.__doc__ is not None
    assert eval_pkg.__doc__ is not None
