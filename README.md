# mimic-complications

Versioned workspace layout:

- `configs/preprocessing/v1.yaml` for preprocessing-specific parameters
- `configs/filtered/v1.yaml` for filtered cohort definitions
- `data/derived/preprocessing/v1/` for saved preprocessing outputs
- `data/derived/filtered/v1/` for saved filtered cohort outputs
- `scripts/prepare_versioned_dataset.py` as the entrypoint scaffold for version-aware runs