"""
schema.py

The standard shape every data source must produce before the detector sees it.

Why this file exists:
    The detector never needs to know where data came from. Today the source is
    the synthetic generator's CSV. Later it could be a Commvault report export or
    the REST API. Each source only has to produce this schema; nothing downstream
    changes. ("The data source is pluggable; here's the interface.")

What load_jobs() does:
    1. Reads the CSV.
    2. Checks that every required column is present.
    3. Converts types (times, numbers) and rejects rows that can't be parsed.
    4. Checks that status values are ones we understand.
    5. Adds a derived "night" column: the backup night each job belongs to.
"""

import pandas as pd

REQUIRED_COLUMNS = [
    "job_id", "client", "agent_type", "site", "media_agent", "backup_level",
    "start_time", "end_time", "duration_min", "size_gb", "status", "error_code",
]
OPTIONAL_COLUMNS = ["error_description"]

VALID_STATUSES = {"Completed", "Completed w/ errors", "Failed"}
VALID_LEVELS = {"Full", "Incremental"}

# A job that starts at 01:30 on Oct 6 belongs to the "night of Oct 5".
# Shifting the start time back 12 hours before taking the date handles that.
NIGHT_ROLLOVER_HOURS = 12


class SchemaError(ValueError):
    """Raised when the input can't be turned into the standard schema."""


def load_jobs(path):
    """Read a job report CSV and return a validated DataFrame."""
    try:
        df = pd.read_csv(path, dtype={"error_code": str, "client": str})
    except Exception as exc:  # unreadable file, wrong format, etc.
        raise SchemaError(f"Could not read {path}: {exc}") from exc
    return validate(df)


def validate(df):
    """Validate and normalize an already-loaded DataFrame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {', '.join(missing)}")

    df = df.copy()
    df["error_code"] = df["error_code"].fillna("").astype(str).str.strip()
    if "error_description" not in df.columns:
        df["error_description"] = ""
    df["error_description"] = df["error_description"].fillna("")

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    df["size_gb"] = pd.to_numeric(df["size_gb"], errors="coerce")

    bad = df[df[["start_time", "end_time", "duration_min", "size_gb"]].isna().any(axis=1)]
    if len(bad):
        raise SchemaError(f"{len(bad)} rows have unparseable times or numbers, "
                          f"e.g. job_id {bad['job_id'].iloc[0]}")

    unknown = set(df["status"]) - VALID_STATUSES
    if unknown:
        raise SchemaError(f"Unknown status values: {sorted(unknown)}")
    unknown = set(df["backup_level"]) - VALID_LEVELS
    if unknown:
        raise SchemaError(f"Unknown backup_level values: {sorted(unknown)}")

    if df["job_id"].duplicated().any():
        raise SchemaError("Duplicate job_id values found")

    df["night"] = (df["start_time"] - pd.Timedelta(hours=NIGHT_ROLLOVER_HOURS)).dt.normalize()
    return df.sort_values(["client", "night"]).reset_index(drop=True)
