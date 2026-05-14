from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ENV_VAR = "SA_SAM2_ARTIFACTS_DIR"
DATA_ENV_VAR = "SA_SAM2_DATA_ROOT"


def get_artifact_root() -> Path:
    env = os.environ.get(ARTIFACTS_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return PROJECT_ROOT.parent / f"{PROJECT_ROOT.name}_artifacts"


def get_data_root() -> Path:
    env = os.environ.get(DATA_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return PROJECT_ROOT / "data"
