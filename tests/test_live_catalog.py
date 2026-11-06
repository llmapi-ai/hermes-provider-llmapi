"""Against the live public catalog (no key): the curated ids must still be live and pickable.

Hermes lists ``fallback_models`` first in the picker, so a retired id is visible to every user.
Runs only with ``LLMAPI_LIVE_CATALOG=1`` (the daily CI job).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
from conftest import PLUGIN_DIR, _install_stubs

pytestmark = pytest.mark.skipif(os.environ.get("LLMAPI_LIVE_CATALOG") != "1", reason="set LLMAPI_LIVE_CATALOG=1")


@pytest.fixture
def live(monkeypatch):
    registered = _install_stubs(monkeypatch, {})
    spec = importlib.util.spec_from_file_location("llmapi_plugin_live", PLUGIN_DIR / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    (profile,) = registered
    picker = profile.fetch_models(timeout=20)
    assert picker, "the live catalog returned no pickable models"
    return module, profile, picker


def test_curated_models_are_live_and_pickable(live):
    module, profile, picker = live
    curated = {*profile.fallback_models, profile.default_aux_model, profile.default_vision_model()}
    assert sorted(curated - set(picker)) == []


def test_vision_model_reads_images(live):
    module, profile, _ = live
    assert profile.model_capabilities[profile.default_vision_model()]["supports_vision"] is True
