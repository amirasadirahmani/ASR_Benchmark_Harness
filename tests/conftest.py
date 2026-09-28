"""Fixtureهای مشترک تست."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_model_manifest(monkeypatch, tmp_path):
    import scripts.setup_models as setup_mod

    monkeypatch.setattr(
        setup_mod,
        "MANIFEST_PATH",
        tmp_path / "model_manifest.json",
    )
    yield
