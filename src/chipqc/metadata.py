"""Load and audit the original OOC spreadsheet without changing its meaning."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


SOURCE_COLUMNS = {
    "imageID": "image_id",
    "cell type": "cell_type",
    "seeding density, cells/ml": "seeding_density",
    "time after seeding, h": "time_after_seeding_h",
    "day": "day",
    "Decision 1/2 (good/bad)": "source_decision",
    "flow rate": "flow_rate",
}
ID_PATTERN = re.compile(r"^\d{6}_\d+$")
LABEL_MAPPING = {1: 1, 2: 0}  # The source header defines 1=good, 2=bad.


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Return normalized records, preserving source fields for audit only.

    `label_good` is 1 for source decision 1 and 0 for source decision 2.
    Invalid source labels stay missing so they cannot silently become a class.
    """
    frame = pd.read_excel(path, sheet_name="Main", dtype={"imageID": "string"})
    absent = set(SOURCE_COLUMNS).difference(frame.columns)
    if absent:
        raise ValueError(f"Missing expected spreadsheet columns: {sorted(absent)}")
    frame = frame.loc[:, list(SOURCE_COLUMNS)].dropna(how="all").rename(columns=SOURCE_COLUMNS)
    frame["image_id"] = frame["image_id"].astype("string").str.strip()
    frame["cell_type"] = frame["cell_type"].astype("string").str.strip()
    frame["group_id"] = frame["image_id"].str.slice(0, 6)
    decision = pd.to_numeric(frame["source_decision"], errors="coerce")
    frame["label_good"] = decision.map(LABEL_MAPPING).astype("Int64")
    return frame.reset_index(drop=True)


def audit_metadata(frame: pd.DataFrame) -> dict:
    """Expose source completeness, labels, IDs, and group leakage risk."""
    id_series = frame["image_id"]
    valid_id = id_series.fillna("").map(lambda value: bool(ID_PATTERN.fullmatch(value)))
    duplicate_id = id_series.duplicated(keep=False) & id_series.notna()
    group_sizes = frame.loc[valid_id].groupby("group_id").size()
    labels = frame["source_decision"].value_counts(dropna=False)
    cells = frame["cell_type"].value_counts(dropna=False)
    by_cell = pd.crosstab(frame["cell_type"], frame["source_decision"])
    valid_groups = frame.loc[valid_id].groupby("group_id")
    numeric_checks = {}
    for column in ("seeding_density", "time_after_seeding_h", "day"):
        original = frame[column]
        normalized = original.astype("string").str.replace(",", "", regex=False)
        numeric = pd.to_numeric(normalized, errors="coerce")
        numeric_checks[column] = {
            "present_count": int(original.notna().sum()),
            "unparseable_present_count": int((original.notna() & numeric.isna()).sum()),
            "negative_count": int((numeric < 0).sum()),
            "min": float(numeric.min()) if numeric.notna().any() else None,
            "max": float(numeric.max()) if numeric.notna().any() else None,
        }
    group_dates = pd.to_datetime(frame.loc[valid_id, "group_id"], format="%y%m%d", errors="coerce")
    return {
        "record_count": int(len(frame)),
        "source_label_mapping": {"1": "good", "2": "bad"},
        "label_mapping_evidence": (
            "Official Zenodo image ZIP preview: test/good/A549/0-1_days/221010_82.png "
            "matches spreadsheet decision 1; test/bad/A549/4+_days/230529_207.png "
            "matches decision 2."
        ),
        "label_counts": {str(key): int(value) for key, value in labels.items()},
        "cell_type_counts": {str(key): int(value) for key, value in cells.items()},
        "label_counts_by_cell_type": {
            str(cell): {str(label): int(count) for label, count in row.items()}
            for cell, row in by_cell.to_dict(orient="index").items()
        },
        "missing_by_field": {column: int(frame[column].isna().sum()) for column in SOURCE_COLUMNS.values()},
        "numeric_field_checks": numeric_checks,
        "flow_rate_raw_variants": {
            str(key): int(value) for key, value in frame["flow_rate"].dropna().value_counts().items()
        },
        "invalid_image_id_count": int((~valid_id).sum()),
        "invalid_group_date_count": int(group_dates.isna().sum()),
        "invalid_image_id_examples": id_series.loc[~valid_id].head(10).fillna("<missing>").tolist(),
        "duplicate_image_id_count": int(duplicate_id.sum()),
        "duplicate_image_id_examples": id_series.loc[duplicate_id].drop_duplicates().head(10).tolist(),
        "invalid_label_count": int(frame["label_good"].isna().sum()),
        "group_key": "first six characters of imageID",
        "group_count": int(group_sizes.size),
        "group_size_min": int(group_sizes.min()) if len(group_sizes) else 0,
        "group_size_median": float(group_sizes.median()) if len(group_sizes) else 0.0,
        "group_size_max": int(group_sizes.max()) if len(group_sizes) else 0,
        "groups_with_one_label": int((valid_groups["label_good"].nunique() == 1).sum()),
        "groups_with_multiple_cell_types": int((valid_groups["cell_type"].nunique() > 1).sum()),
        "warning": "Rows from one six-character group must stay in the same data split.",
        "model_input_note": "Only image pixel features enter the baseline; spreadsheet fields are audit/split/stratification metadata.",
    }


def require_trainable_metadata(frame: pd.DataFrame) -> None:
    """Fail closed on IDs or labels that would make matching/splits ambiguous."""
    report = audit_metadata(frame)
    problems = [
        key for key in ("invalid_image_id_count", "duplicate_image_id_count", "invalid_label_count")
        if report[key]
    ]
    if problems:
        raise ValueError(f"Metadata cannot be prepared safely: {', '.join(problems)}")
