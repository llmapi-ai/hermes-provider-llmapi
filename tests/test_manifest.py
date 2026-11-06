"""The manifest Hermes reads at install time, and the network-use promise."""

from __future__ import annotations

import ast
import re

import yaml
from conftest import PLUGIN_DIR, ROOT


def test_manifest():
    manifest = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
    assert manifest["name"] == "llmapi"
    assert manifest["kind"] == "model-provider"
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(manifest["version"]))
    assert manifest["requires_hermes"] == ">=0.22.6"  # api_mode_for_model first shipped there
    (env,) = manifest["requires_env"]
    assert env == {"name": "LLMAPI_API_KEY", "url": "https://app.llmapi.ai/dashboard/api-keys", "secret": True}


def test_versions_agree(plugin):
    manifest = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert plugin.__version__ == manifest["version"]
    assert f'version = "{plugin.__version__}"' in pyproject


def test_imports_only_public_hermes_modules():
    tree = ast.parse((PLUGIN_DIR / "__init__.py").read_text())
    public = {"providers", "providers.base", "agent.account_usage", "agent.usage_pricing", "agent.secret_scope"}
    stdlib = {
        "__future__",
        "dataclasses",
        "json",
        "logging",
        "re",
        "threading",
        "time",
        "urllib.error",
        "urllib.request",
        "urllib.parse",
        "datetime",
        "decimal",
        "typing",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module in public | stdlib, node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in stdlib, alias.name


def test_no_file_or_process_access():
    source = (PLUGIN_DIR / "__init__.py").read_text()
    for forbidden in (r"(?<![\w.])open\(", r"\bsubprocess\b", r"\bos\.", r"\bPath\(", r"\bsocket\b"):
        assert not re.search(forbidden, source), forbidden
