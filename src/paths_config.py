"""Load config.yaml and resolve DCEC script paths."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_CFG_PATH = _REPO / "config.yaml"


def load_config():
    with open(_CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    return _REPO


def dcec_rgb() -> Path:
    return Path(load_config()["torch_dcec_rgb"])


def script(name: str) -> Path:
    return dcec_rgb() / name


def analysis_out(method: str) -> Path:
    p = repo_root() / "analysis" / "outputs" / method
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_dir(tag: str) -> Path:
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d")
    p = repo_root() / "logs" / f"{stamp}-{tag}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def venv_python() -> Path:
    py = dcec_rgb() / "venv" / "Scripts" / "python.exe"
    if not py.is_file():
        raise FileNotFoundError(f"Expected venv Python: {py}")
    return py
