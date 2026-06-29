from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mimic_complications.paths import ensure_versioned_dirs, filtered_config_path, preprocessing_config_path  # noqa: E402
from mimic_complications.pipeline import DEFAULT_MISSING_THRESHOLD, run_pipeline  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build versioned MIMIC preprocessing and filtered datasets.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "mimic")
    parser.add_argument("--preprocessing-version", default="v1")
    parser.add_argument("--filtered-version", default="v1")
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessing_config = load_config(preprocessing_config_path(args.preprocessing_version))
    filtered_config = load_config(filtered_config_path(args.filtered_version))

    preprocessing_dir, filtered_dir = ensure_versioned_dirs(args.preprocessing_version, args.filtered_version)
    missing_threshold = (
        preprocessing_config.get("missingness", {}).get("threshold")
        or filtered_config.get("criteria", {}).get("missing_rate_threshold")
        or DEFAULT_MISSING_THRESHOLD
    )

    result = run_pipeline(
        data_root=args.data_root,
        preprocessing_version=args.preprocessing_version,
        filtered_version=args.filtered_version,
        preprocessing_output_dir=preprocessing_dir,
        filtered_output_dir=filtered_dir,
        missing_threshold=float(missing_threshold),
        chunksize=args.chunksize,
    )

    print("Versioned dataset build complete.")
    print(f"Preprocessing config: {preprocessing_config.get('name', args.preprocessing_version)}")
    print(f"Filtered config: {filtered_config.get('name', args.filtered_version)}")
    print(f"Sepsis stays: {result.summary['sepsis_stays']}")
    print(f"Sepsis patients: {result.summary['sepsis_patients']}")
    print(f"Rows saved: {result.summary['retained_rows']}")
    print(f"Columns saved: {result.summary['retained_columns']}")


if __name__ == "__main__":
    main()
