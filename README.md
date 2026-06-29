# mimic-complications

Versioned workspace layout:

- `configs/preprocessing/v1.yaml` for preprocessing-specific parameters
- `configs/filtered/v1.yaml` for filtered cohort definitions
- `data/derived/preprocessing/v1/` for saved preprocessing outputs
- `data/derived/filtered/v1/` for saved filtered cohort outputs
- `scripts/prepare_versioned_dataset.py` as the entrypoint scaffold for version-aware runs

Run the first version with:

```powershell
python scripts/prepare_versioned_dataset.py --preprocessing-version v1 --filtered-version v1
```

This writes the processed cohort to `data/derived/preprocessing/v1/dataset.csv`, the filtered version to `data/derived/filtered/v1/dataset.csv`, and a JSON summary to `data/derived/filtered/v1/summary.json`.