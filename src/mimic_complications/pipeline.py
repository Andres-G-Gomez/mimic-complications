from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEPSIS_CODE_PREFIXES = (
    "038",
    "99591",
    "99592",
    "78552",
    "A40",
    "A41",
    "R6520",
    "R6521",
)

DEFAULT_WINDOW_HOURS = 24
DEFAULT_MISSING_THRESHOLD = 0.70


@dataclass(frozen=True)
class PipelineResult:
    cohort: pd.DataFrame
    processed: pd.DataFrame
    summary: dict[str, object]


FEATURE_PATTERNS = {
    "chart": {
        "heart_rate": [r"^Heart Rate$", r"^HR$"],
        "systolic_bp": [r"systolic blood pressure", r"^Arterial Blood Pressure systolic$", r"^ABPs?$"],
        "diastolic_bp": [r"diastolic blood pressure", r"^Arterial Blood Pressure diastolic$", r"^ABPd?$"],
        "mean_arterial_pressure": [r"mean arterial pressure", r"^MAP$", r"arterial blood pressure mean"],
        "respiratory_rate": [r"respiratory rate", r"^RR$"],
        "oxygen_saturation": [r"oxygen saturation", r"^SpO2$", r"^O2 saturation$"],
        "temperature": [r"temperature", r"^Temp$", r"^Body Temperature$"],
        "glucose": [r"^Glucose$", r"finger stick glucose"],
        "sofa_score": [r"^SOFA Score$"],
        "apache_ii_score": [r"^APACHE II$", r"APACHE II"],
        "apache_iv_score": [r"APACHE IV", r"ApacheIV", r"APS", r"Apache IV .*score"],
        "saps_score": [r"SAPS"],
    },
    "lab": {
        "lactate": [r"lactate"],
        "creatinine": [r"creatinine"],
        "bilirubin_total": [r"bilirubin.*total", r"total bilirubin"],
        "hemoglobin": [r"^hemoglobin$", r"^hgb$"],
        "hematocrit": [r"^hematocrit$", r"^hct$"],
        "platelets": [r"platelets"],
        "wbc": [r"white blood cell", r"^wbc$"],
        "sodium": [r"^sodium$"],
        "potassium": [r"^potassium$"],
        "chloride": [r"^chloride$"],
        "bicarbonate": [r"bicarbonate", r"co2"],
        "anion_gap": [r"anion gap"],
        "glucose": [r"^glucose$"],
        "bun": [r"blood urea nitrogen", r"^bun$"],
        "albumin": [r"^albumin$"],
        "ast": [r"^ast$", r"aspartate aminotransferase"],
        "alt": [r"^alt$", r"alanine aminotransferase"],
        "inr": [r"international normalized ratio", r"^inr$"],
        "pt": [r"prothrombin time", r"^pt$"],
        "ptt": [r"partial thromboplastin time", r"^ptt$"],
        "calcium": [r"^calcium$"],
        "magnesium": [r"^magnesium$"],
        "phosphate": [r"phosphate"],
        "ph": [r"^ph$"],
        "paco2": [r"paco2", r"co2 arterial"],
        "pao2": [r"pao2"],
        "base_excess": [r"base excess"],
    },
    "output": {
        "urine_output": [r"urine", r"void", r"foley", r"condom cath", r"straight cath", r"suprapubic", r"nephrostomy"],
    },
}

DEMOGRAPHIC_CATEGORICAL_COLUMNS = [
    "gender",
    "anchor_year_group",
    "admission_type",
    "admission_location",
    "discharge_location",
    "insurance",
    "language",
    "marital_status",
    "race",
    "first_careunit",
    "last_careunit",
]


def read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer", low_memory=False, **kwargs)


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of the candidate files exist: {', '.join(str(path) for path in candidates)}")


def _normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()


def load_reference_tables(data_root: Path) -> dict[str, pd.DataFrame]:
    hosp_root = data_root / "hosp"
    icu_root = data_root / "icu"

    tables = {
        "patients": read_csv(
            hosp_root / "patients.csv.gz",
            usecols=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"],
        ),
        "admissions": read_csv(
            hosp_root / "admissions.csv.gz",
            usecols=[
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "deathtime",
                "admission_type",
                "admission_location",
                "discharge_location",
                "insurance",
                "language",
                "marital_status",
                "race",
                "hospital_expire_flag",
            ],
        ),
        "icustays": read_csv(icu_root / "icustays.csv", parse_dates=["intime", "outtime"]),
        "diagnoses": read_csv(
            resolve_existing_path(hosp_root / "diagnoses_icd.csv.gz", hosp_root / "diagnoses_icd.csv"),
            usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
            dtype={"icd_code": str, "icd_version": str},
        ),
        "d_icd_diagnoses": read_csv(
            resolve_existing_path(hosp_root / "d_icd_diagnoses.csv.gz", hosp_root / "d_icd_diagnoses.csv"),
            usecols=["icd_code", "icd_version", "long_title"],
            dtype={"icd_code": str, "icd_version": str},
        ),
        "d_items": read_csv(icu_root / "d_items.csv"),
        "d_labitems": read_csv(hosp_root / "d_labitems.csv.gz"),
    }
    return tables


def identify_sepsis_admissions(diagnoses: pd.DataFrame, d_icd_diagnoses: pd.DataFrame) -> pd.Index:
    merged = diagnoses.merge(d_icd_diagnoses, on=["icd_code", "icd_version"], how="left")
    code_series = merged["icd_code"].map(_normalize_code)
    title_series = merged["long_title"].fillna("").str.lower()
    code_mask = code_series.str.startswith(SEPSIS_CODE_PREFIXES)
    title_mask = title_series.str.contains(r"sepsis|septicemia|septic shock", regex=True, na=False)
    return pd.Index(merged.loc[code_mask | title_mask, "hadm_id"].dropna().unique())


def build_cohort(
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
    sepsis_hadm_ids: Iterable[int],
) -> pd.DataFrame:
    cohort = (
        icustays.merge(admissions, on=["subject_id", "hadm_id"], how="inner", suffixes=("", "_adm"))
        .merge(patients, on="subject_id", how="left", suffixes=("", "_pat"))
    )

    cohort = cohort.loc[
        cohort["hadm_id"].isin(pd.Index(sepsis_hadm_ids))
        & cohort["anchor_age"].ge(18)
        & cohort["los"].gt(1)
        & cohort["dischtime"].notna()
        & cohort["discharge_location"].notna()
    ].copy()

    cohort["intime"] = pd.to_datetime(cohort["intime"])
    cohort["outtime"] = pd.to_datetime(cohort["outtime"])
    cohort["admittime"] = pd.to_datetime(cohort["admittime"])
    cohort["dischtime"] = pd.to_datetime(cohort["dischtime"])
    cohort["deathtime"] = pd.to_datetime(cohort["deathtime"])
    cohort["icu_los_days"] = cohort["los"]
    cohort["hospital_los_days"] = (cohort["dischtime"] - cohort["admittime"]).dt.total_seconds() / 86400.0
    cohort["sepsis_flag"] = 1

    return cohort.drop_duplicates(subset=["stay_id"])


def _select_dictionary_items(dictionary: pd.DataFrame, patterns: dict[str, list[str]], label_column: str = "label") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    label_values = dictionary[label_column].fillna("").astype(str)
    numeric_dictionary = dictionary.copy()
    if "param_type" in numeric_dictionary.columns:
        numeric_dictionary = numeric_dictionary.loc[numeric_dictionary["param_type"].fillna("").str.contains("numeric", case=False, na=False)]
        label_values = numeric_dictionary[label_column].fillna("").astype(str)

    for feature_name, regexes in patterns.items():
        regex = "|".join(f"(?:{expr})" for expr in regexes)
        mask = label_values.str.contains(regex, case=False, regex=True, na=False)
        for _, row in numeric_dictionary.loc[mask].iterrows():
            rows.append(
                {
                    "itemid": int(row["itemid"]),
                    "feature_name": feature_name,
                    "label": row[label_column],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["itemid", "feature_name", "label"])

    return pd.DataFrame(rows).drop_duplicates(subset=["itemid", "feature_name"])


def build_feature_catalog(d_items: pd.DataFrame, d_labitems: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "chart": _select_dictionary_items(d_items, FEATURE_PATTERNS["chart"]),
        "lab": _select_dictionary_items(d_labitems, FEATURE_PATTERNS["lab"]),
        "output": _select_dictionary_items(d_items, FEATURE_PATTERNS["output"]),
    }


def _aggregate_event_chunks(
    event_path: Path,
    *,
    cohort: pd.DataFrame,
    join_key: str,
    time_col: str,
    itemid_to_feature: pd.Series,
    value_col: str,
    chunksize: int,
    usecols: list[str],
) -> pd.DataFrame:
    cohort = cohort.loc[:, ~cohort.columns.duplicated()].copy()
    base_columns = [join_key, "intime"]
    if join_key != "stay_id":
        base_columns.insert(0, "stay_id")
    cohort_window = cohort[base_columns].copy()
    cohort_window["window_end"] = cohort_window["intime"] + pd.Timedelta(hours=DEFAULT_WINDOW_HOURS)

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        event_path,
        compression="infer",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        if chunk.empty:
            continue
        chunk[time_col] = pd.to_datetime(chunk[time_col], errors="coerce")
        chunk[value_col] = pd.to_numeric(chunk[value_col], errors="coerce")
        chunk = chunk.loc[chunk["itemid"].isin(itemid_to_feature.index)]
        if chunk.empty:
            continue

        merged = chunk.merge(cohort_window, on=join_key, how="inner")
        merged = merged.loc[(merged[time_col] >= merged["intime"]) & (merged[time_col] <= merged["window_end"])]
        if merged.empty:
            continue

        merged["feature_name"] = merged["itemid"].map(itemid_to_feature)
        merged = merged.dropna(subset=["feature_name", value_col])
        if merged.empty:
            continue

        frames.append(merged[["stay_id", "feature_name", value_col]])

    if not frames:
        return pd.DataFrame(index=pd.Index(cohort["stay_id"], name="stay_id"))

    stacked = pd.concat(frames, ignore_index=True)
    return stacked.groupby(["stay_id", "feature_name"], observed=True)[value_col].mean().unstack("feature_name")


def extract_first_24h_features(
    data_root: Path,
    cohort: pd.DataFrame,
    feature_catalog: dict[str, pd.DataFrame],
    *,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    icu_root = data_root / "icu"
    hosp_root = data_root / "hosp"

    chart_items = feature_catalog["chart"]
    lab_items = feature_catalog["lab"]
    output_items = feature_catalog["output"]

    chart_map = chart_items.set_index("itemid")["feature_name"] if not chart_items.empty else pd.Series(dtype=str)
    lab_map = lab_items.set_index("itemid")["feature_name"] if not lab_items.empty else pd.Series(dtype=str)
    output_map = output_items.set_index("itemid")["feature_name"] if not output_items.empty else pd.Series(dtype=str)

    chart_features = _aggregate_event_chunks(
        icu_root / "chartevents.csv.gz",
        cohort=cohort,
        join_key="stay_id",
        time_col="charttime",
        itemid_to_feature=chart_map,
        value_col="valuenum",
        chunksize=chunksize,
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
    )

    output_features = _aggregate_event_chunks(
        resolve_existing_path(icu_root / "outputevents.csv.gz", icu_root / "outputevents.csv"),
        cohort=cohort,
        join_key="stay_id",
        time_col="charttime",
        itemid_to_feature=output_map,
        value_col="value",
        chunksize=chunksize,
        usecols=["stay_id", "charttime", "itemid", "value"],
    )

    lab_features = _aggregate_event_chunks(
        hosp_root / "labevents.csv.gz",
        cohort=cohort,
        join_key="hadm_id",
        time_col="charttime",
        itemid_to_feature=lab_map,
        value_col="valuenum",
        chunksize=chunksize,
        usecols=["hadm_id", "charttime", "itemid", "valuenum"],
    )

    features = pd.concat([chart_features, output_features, lab_features], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    features = features.reindex(cohort["stay_id"].values)

    return features


def _add_demographics(cohort: pd.DataFrame) -> pd.DataFrame:
    demographic_columns = [
        "stay_id",
        "subject_id",
        "hadm_id",
        "gender",
        "anchor_age",
        "anchor_year",
        "anchor_year_group",
        "race",
        "marital_status",
        "insurance",
        "language",
        "admission_type",
        "admission_location",
        "discharge_location",
        "first_careunit",
        "last_careunit",
        "icu_los_days",
        "hospital_los_days",
        "hospital_expire_flag",
        "sepsis_flag",
    ]
    cols = [col for col in demographic_columns if col in cohort.columns]
    return cohort[cols].copy()


def _impute_missing_values(df: pd.DataFrame, categorical_columns: list[str], identifier_columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        if column in identifier_columns:
            continue
        if column in categorical_columns or result[column].dtype == object:
            result[column] = result[column].astype("string").fillna("null")
            continue
        result[column] = pd.to_numeric(result[column], errors="coerce")
        median_value = result[column].median(skipna=True)
        result[column] = result[column].fillna(median_value)
    return result


def run_pipeline(
    *,
    data_root: Path,
    preprocessing_version: str,
    filtered_version: str,
    preprocessing_output_dir: Path,
    filtered_output_dir: Path,
    missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
    chunksize: int = 250_000,
) -> PipelineResult:
    tables = load_reference_tables(data_root)
    sepsis_hadm_ids = identify_sepsis_admissions(tables["diagnoses"], tables["d_icd_diagnoses"])
    cohort = build_cohort(tables["patients"], tables["admissions"], tables["icustays"], sepsis_hadm_ids)

    feature_catalog = build_feature_catalog(tables["d_items"], tables["d_labitems"])
    features = extract_first_24h_features(data_root, cohort, feature_catalog, chunksize=chunksize)

    demographics = _add_demographics(cohort)
    processed = demographics.merge(features.reset_index().rename(columns={"index": "stay_id"}), on="stay_id", how="left")
    processed = processed.loc[:, ~processed.columns.duplicated()]

    identifier_columns = ["stay_id", "subject_id", "hadm_id"]
    candidate_columns = [column for column in processed.columns if column not in identifier_columns]
    missing_rates = processed[candidate_columns].isna().mean()
    dropped_columns = [column for column, rate in missing_rates.items() if rate >= missing_threshold]
    retained_columns = identifier_columns + [column for column in candidate_columns if column not in dropped_columns]
    processed = processed[retained_columns].copy()

    processed = _impute_missing_values(processed, DEMOGRAPHIC_CATEGORICAL_COLUMNS, identifier_columns)

    preprocessing_output_dir.mkdir(parents=True, exist_ok=True)
    filtered_output_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_path = preprocessing_output_dir / "dataset.csv"
    filtered_path = filtered_output_dir / "dataset.csv"
    summary_path = filtered_output_dir / "summary.json"

    processed.to_csv(preprocessing_path, index=False)
    processed.to_csv(filtered_path, index=False)

    summary = {
        "preprocessing_version": preprocessing_version,
        "filtered_version": filtered_version,
        "sepsis_patients": int(cohort["subject_id"].nunique()),
        "sepsis_stays": int(cohort["stay_id"].nunique()),
        "retained_rows": int(processed.shape[0]),
        "retained_columns": int(processed.shape[1]),
        "dropped_columns_missing_threshold": dropped_columns,
        "missing_threshold": missing_threshold,
        "feature_count_before_filter": int(features.shape[1]),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return PipelineResult(cohort=cohort, processed=processed, summary=summary)