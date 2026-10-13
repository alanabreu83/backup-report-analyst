#!/usr/bin/env python3
"""
detector.py

Finds problems in a backup job report using plain code (pandas), not an LLM.

Why code and not an LLM for this step:
    LLMs are unreliable at counting and arithmetic across hundreds of rows. Code
    is exact, fast, cheap, and testable. The LLM's job (Days 5-6) is to EXPLAIN
    these findings, not to find them.

What it looks for, in this order (order matters, see "consumed" below):
    1. media_agent_outage   many jobs on one media agent failing the same night
    2. repeat_failure       a client failing 3+ nights in a row
    3. missed_backup        a client with no job on a night it should have run
    4. size_drop            a "successful" job that backed up far less than normal
    5. size_spike           an incremental far larger than normal
    6. duration_spike       a job far slower than normal, not explained by size
    Everything else that failed or warned once is listed as "one_off" for
    information only. A good detector should not escalate a single blip.

"consumed" jobs:
    Once a job is explained by a bigger finding (e.g. it failed because its
    media agent was down), it is not reported again as a separate problem.
    That's how 10 failures become ONE root cause instead of 10 alerts.

How "normal" is measured:
    Each job is compared with the MEDIAN of the same client's other successful
    jobs at the same backup level (fulls vs. fulls, incrementals vs.
    incrementals). The median ignores the occasional outlier, and comparing a
    client only with itself means a 2 TB database and a 50 GB file server are
    each judged by their own history.

Usage:
    python detector.py backup_jobs.csv                    # prints a summary
    python detector.py backup_jobs.csv --out findings.json
"""

import argparse
import json
from collections import Counter

import pandas as pd

from schema import load_jobs

# ---------------------------------------------------------------------------
# Thresholds. Every number that decides "anomaly or not" lives here, so it can
# be explained and tuned in one place.
# ---------------------------------------------------------------------------
MA_OUTAGE_MIN_FAILED = 3        # at least this many failures on one media agent...
MA_OUTAGE_MIN_FAIL_RATIO = 0.5  # ...and at least half of its jobs that night
REPEAT_FAILURE_MIN_NIGHTS = 3   # consecutive failed nights to call it ongoing
MIN_BASELINE_JOBS = 5           # need this much history to judge "normal"
SIZE_DROP_RATIO = 0.30          # completed, but <= 30% of normal size
SIZE_SPIKE_RATIO = 3.0          # incremental >= 3x normal size
DURATION_SPIKE_RATIO = 2.5      # >= 2.5x normal duration...
DURATION_SPIKE_MIN_EXTRA = 10   # ...and at least 10 minutes longer (ignores 4 -> 11 min blips)...
DURATION_SIZE_EXPLAINED = 2.0   # ...unless size also grew >= 2x (then size explains it)

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2}
SUCCESS_STATUSES = {"Completed", "Completed w/ errors"}


def _date(ts):
    return ts.strftime("%Y-%m-%d")


def _ids(series):
    return sorted(int(x) for x in series)


# ---------------------------------------------------------------------------
# Individual checks. Each returns a list of finding dicts and the job IDs it
# has explained.
# ---------------------------------------------------------------------------

def find_media_agent_outages(df):
    findings, consumed = [], set()
    for (ma, night), group in df.groupby(["media_agent", "night"]):
        failed = group[group["status"] == "Failed"]
        if len(failed) < MA_OUTAGE_MIN_FAILED:
            continue
        ratio = len(failed) / len(group)
        if ratio < MA_OUTAGE_MIN_FAIL_RATIO:
            continue
        codes = Counter(failed["error_code"])
        findings.append({
            "type": "media_agent_outage",
            "severity": "critical",
            "clients": sorted(failed["client"]),
            "media_agent": ma,
            "site": group["site"].iloc[0],
            "nights": [_date(night)],
            "job_ids": _ids(failed["job_id"]),
            "evidence": {
                "failed_jobs": len(failed),
                "total_jobs_on_media_agent": len(group),
                "fail_ratio": round(ratio, 2),
                "error_codes": dict(codes.most_common()),
            },
            "summary": (f"{len(failed)} of {len(group)} jobs on {ma} failed on "
                        f"{_date(night)}; most common error {codes.most_common(1)[0][0]}. "
                        f"Likely one shared root cause on the media agent."),
        })
        consumed.update(failed["job_id"])
    return findings, consumed


def find_repeat_failures(df, report_nights, consumed):
    findings, newly = [], set()
    night_index = {n: i for i, n in enumerate(report_nights)}
    last_night = report_nights[-1]

    for client, group in df.groupby("client"):
        failed = group[(group["status"] == "Failed") & ~group["job_id"].isin(consumed)]
        if failed.empty:
            continue
        failed = failed.sort_values("night")
        # Split failures into runs of consecutive report nights
        runs, current = [], [failed.iloc[0]]
        for _, row in failed.iloc[1:].iterrows():
            if night_index[row["night"]] == night_index[current[-1]["night"]] + 1:
                current.append(row)
            else:
                runs.append(current)
                current = [row]
        runs.append(current)

        for run in runs:
            if len(run) < REPEAT_FAILURE_MIN_NIGHTS:
                continue
            run_df = pd.DataFrame(run)
            still_failing = run_df["night"].max() == last_night
            codes = Counter(run_df["error_code"])
            findings.append({
                "type": "repeat_failure",
                "severity": "critical" if still_failing else "high",
                "clients": [client],
                "media_agent": run_df["media_agent"].iloc[0],
                "site": run_df["site"].iloc[0],
                "nights": [_date(n) for n in run_df["night"]],
                "job_ids": _ids(run_df["job_id"]),
                "evidence": {
                    "consecutive_failed_nights": len(run_df),
                    "still_failing_on_last_night": bool(still_failing),
                    "error_codes": dict(codes.most_common()),
                },
                "summary": (f"{client} failed {len(run_df)} nights in a row "
                            f"({_date(run_df['night'].min())} to {_date(run_df['night'].max())}) "
                            f"with {codes.most_common(1)[0][0]}"
                            f"{'; still failing as of the last night' if still_failing else ''}."),
            })
            newly.update(run_df["job_id"])
    return findings, newly


def find_missed_backups(df, report_nights):
    """A client that has run before, but has no job at all on a later night."""
    findings = []
    for client, group in df.groupby("client"):
        ran = set(group["night"])
        first_seen = group["night"].min()
        for night in report_nights:
            if night < first_seen or night in ran:
                continue
            last_job = group[group["night"] < night].sort_values("night").iloc[-1]
            findings.append({
                "type": "missed_backup",
                "severity": "high",
                "clients": [client],
                "media_agent": group["media_agent"].iloc[0],
                "site": group["site"].iloc[0],
                "nights": [_date(night)],
                "job_ids": [],
                "evidence": {
                    "previous_job_id": int(last_job["job_id"]),
                    "previous_job_night": _date(last_job["night"]),
                    "previous_job_status": last_job["status"],
                },
                "summary": (f"No backup job ran for {client} on {_date(night)}. "
                            f"Status-based failure reports would not show this."),
            })
    return findings


def find_size_and_duration_anomalies(df, consumed):
    """Compare each successful job with the same client's normal at the same level."""
    findings, newly = [], set()
    # Baseline = jobs that ran to the end (warnings included). Failed jobs are excluded
    # because their partial sizes and durations would drag "normal" down.
    baseline_pool = df[df["status"].isin(SUCCESS_STATUSES)]

    candidates = df[df["status"].isin(SUCCESS_STATUSES) & ~df["job_id"].isin(consumed)]
    for _, job in candidates.iterrows():
        others = baseline_pool[(baseline_pool["client"] == job["client"])
                               & (baseline_pool["backup_level"] == job["backup_level"])
                               & (baseline_pool["job_id"] != job["job_id"])]
        if len(others) < MIN_BASELINE_JOBS:
            continue  # not enough history to say what's normal

        median_size = others["size_gb"].median()
        median_dur = others["duration_min"].median()
        size_ratio = job["size_gb"] / median_size if median_size else float("inf")
        dur_ratio = job["duration_min"] / median_dur if median_dur else float("inf")
        level = job["backup_level"].lower()

        base = {
            "clients": [job["client"]],
            "media_agent": job["media_agent"],
            "site": job["site"],
            "nights": [_date(job["night"])],
            "job_ids": [int(job["job_id"])],
            "evidence": {
                "backup_level": job["backup_level"],
                "status": job["status"],
                "size_gb": float(job["size_gb"]),
                "normal_size_gb": round(float(median_size), 1),
                "size_ratio": round(float(size_ratio), 2),
                "duration_min": float(job["duration_min"]),
                "normal_duration_min": round(float(median_dur), 1),
                "duration_ratio": round(float(dur_ratio), 2),
                "baseline_jobs": len(others),
            },
        }

        if size_ratio <= SIZE_DROP_RATIO:
            findings.append({**base, "type": "size_drop", "severity": "high",
                "summary": (f"{job['client']} reported {job['status']} but backed up "
                            f"{job['size_gb']:.1f} GB vs. a normal {median_size:.1f} GB "
                            f"({size_ratio:.0%} of normal {level}). Possible silent failure.")})
            newly.add(job["job_id"])
        elif job["backup_level"] == "Incremental" and size_ratio >= SIZE_SPIKE_RATIO:
            findings.append({**base, "type": "size_spike", "severity": "high",
                "summary": (f"{job['client']} incremental was {job['size_gb']:.1f} GB vs. a "
                            f"normal {median_size:.1f} GB ({size_ratio:.1f}x). Abnormal change "
                            f"rate; check for mass data change or ransomware encryption.")})
            newly.add(job["job_id"])
        elif (dur_ratio >= DURATION_SPIKE_RATIO
              and job["duration_min"] - median_dur >= DURATION_SPIKE_MIN_EXTRA
              and size_ratio < DURATION_SIZE_EXPLAINED):
            findings.append({**base, "type": "duration_spike", "severity": "medium",
                "summary": (f"{job['client']} {level} took {job['duration_min']:.0f} min vs. a "
                            f"normal {median_dur:.0f} min ({dur_ratio:.1f}x) with no matching "
                            f"growth in size. Check throughput, network or media agent load.")})
            newly.add(job["job_id"])
    return findings, newly


def list_one_off_issues(df, consumed):
    """Failures and warnings not explained by any finding. Informational only."""
    rest = df[~df["job_id"].isin(consumed) & (df["status"] != "Completed")]
    return [{
        "job_id": int(r["job_id"]),
        "client": r["client"],
        "night": _date(r["night"]),
        "status": r["status"],
        "error_code": r["error_code"],
        "error_description": r["error_description"],
    } for _, r in rest.sort_values(["night", "client"]).iterrows()]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def detect(df):
    report_nights = sorted(df["night"].unique())
    consumed = set()

    outages, used = find_media_agent_outages(df)
    consumed |= used
    repeats, used = find_repeat_failures(df, report_nights, consumed)
    consumed |= used
    missed = find_missed_backups(df, report_nights)
    sized, used = find_size_and_duration_anomalies(df, consumed)
    consumed |= used

    findings = sorted(outages + repeats + missed + sized,
                      key=lambda f: (SEVERITY_ORDER[f["severity"]], f["nights"][0], f["clients"][0]))
    for i, f in enumerate(findings, start=1):
        f["id"] = f"F{i:02d}"
    findings = [{"id": f.pop("id"), **f} for f in findings]

    status_counts = df["status"].value_counts().to_dict()
    return {
        "report": {
            "first_night": _date(report_nights[0]),
            "last_night": _date(report_nights[-1]),
            "nights": len(report_nights),
            "clients": int(df["client"].nunique()),
            "total_jobs": int(len(df)),
            "status_counts": {k: int(v) for k, v in status_counts.items()},
        },
        "findings": findings,
        "one_off_issues": list_one_off_issues(df, consumed),
        "thresholds": {
            "ma_outage_min_failed": MA_OUTAGE_MIN_FAILED,
            "ma_outage_min_fail_ratio": MA_OUTAGE_MIN_FAIL_RATIO,
            "repeat_failure_min_nights": REPEAT_FAILURE_MIN_NIGHTS,
            "min_baseline_jobs": MIN_BASELINE_JOBS,
            "size_drop_ratio": SIZE_DROP_RATIO,
            "size_spike_ratio": SIZE_SPIKE_RATIO,
            "duration_spike_ratio": DURATION_SPIKE_RATIO,
            "duration_spike_min_extra_min": DURATION_SPIKE_MIN_EXTRA,
        },
    }


def main():
    p = argparse.ArgumentParser(description="Detect anomalies in a backup job report.")
    p.add_argument("csv", help="Path to the job report CSV")
    p.add_argument("--out", help="Write findings JSON to this path")
    args = p.parse_args()

    result = detect(load_jobs(args.csv))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)

    r = result["report"]
    print(f"Report {r['first_night']} to {r['last_night']}: {r['total_jobs']} jobs, "
          f"{r['clients']} clients, statuses {r['status_counts']}")
    print(f"{len(result['findings'])} findings, "
          f"{len(result['one_off_issues'])} one-off issues (informational)\n")
    for f in result["findings"]:
        print(f"  {f['id']}  [{f['severity']:<8}] {f['type']:<20} {f['summary']}")
    if args.out:
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
