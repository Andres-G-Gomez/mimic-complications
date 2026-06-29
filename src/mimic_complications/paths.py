from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DERIVED_ROOT = PROJECT_ROOT / "data" / "derived"
CONFIG_ROOT = PROJECT_ROOT / "configs"


def preprocessing_output_dir(version: str) -> Path:
    return DERIVED_ROOT / "preprocessing" / version


def filtered_output_dir(version: str) -> Path:
    return DERIVED_ROOT / "filtered" / version


def preprocessing_config_path(version: str) -> Path:
    return CONFIG_ROOT / "preprocessing" / f"{version}.yaml"


def filtered_config_path(version: str) -> Path:
    return CONFIG_ROOT / "filtered" / f"{version}.yaml"


def ensure_versioned_dirs(preprocessing_version: str, filtered_version: str) -> tuple[Path, Path]:
    preprocessing_dir = preprocessing_output_dir(preprocessing_version)
    filtered_dir = filtered_output_dir(filtered_version)
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)
    return preprocessing_dir, filtered_dir
