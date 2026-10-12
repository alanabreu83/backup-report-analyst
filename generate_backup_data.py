#!/usr/bin/env python3
"""
generate_backup_data.py

Generates a synthetic, Commvault-style nightly backup job report (CSV) plus a
separate answer key (JSON) listing the anomalies that were deliberately planted.

Why this exists:
    The Backup Report Analyst must be *measurable*. The detector never reads the
    answer key. After it runs, you compare its findings to the key and report
    results like "caught 12 of 13 planted anomalies, 1 false positive."

What is in the data:
    1. Normal jobs: one job per client per night, with a weekly full backup and
       incrementals the other nights. Size and duration vary realistically.
    2. Background noise (NOT anomalies): ~2% one-off failures and ~3%
       "Completed w/ errors". A good detector should not panic over these.
    3. Planted anomalies (13 by default):
         - repeat_failure       a client failing 3+ nights in a row
         - duration_spike       job completes but runs 4-6x longer than normal
         - size_drop            job "Completed" but backed up ~90% less data
         - size_spike           incremental 5-10x normal size (possible ransomware)
         - missed_backup        no job at all for a client on a night
         - media_agent_outage   every job on one media agent fails the same night

Standard library only. Usage:
    python generate_backup_data.py                       # seed 42, 14 nights
    python generate_backup_data.py --seed 7
    python generate_backup_data.py --nights 21 --out-dir data
"""

import argparse
import csv
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment definition
# ---------------------------------------------------------------------------

SITES = {
    "US-EAST":    {"code": "use", "media_agents": ["MA-USE-01", "MA-USE-02"]},
    "US-CENTRAL": {"code": "usc", "media_agents": ["MA-USC-01", "MA-USC-02"]},
    "UK-LONDON":  {"code": "ukl", "media_agents": ["MA-UKL-01", "MA-UKL-02"]},
}

# agent type -> (client name prefix, full size GB range,
#                throughput GB/hour range, daily change rate range, weight)
AGENT_PROFILES = {
    "VMware":              ("vmw",  (200, 1500), (300, 600), (0.02, 0.06), 30),
    "SQL Server":          ("sql",  (100, 800),  (250, 500), (0.05, 0.12), 15),
    "Oracle":              ("ora",  (300, 2000), (200, 450), (0.04, 0.10), 10),
    "Windows File System": ("win",  (50, 600),   (120, 300), (0.01, 0.04), 20),
    "Linux File System":   ("lnx",  (50, 500),   (150, 350), (0.01, 0.04), 15),
    "SAP HANA":            ("hana", (500, 3000), (300, 600), (0.05, 0.10), 10),
}

# Placeholder error codes. TODO: replace with real codes and descriptions you
# know from experience. Keep them in sync with the knowledge base (Days 7-8).
ERROR_CATALOG = {
    "Error Code: 101": ("Failed", "Unable to connect to client; communication services not responding"),
    "Error Code: 102": ("Failed", "Media agent unreachable or offline"),
    "Error Code: 103": ("Failed", "VSS snapshot creation failed on client"),
    "Error Code: 104": ("Failed", "Database log backup chain broken; full backup required"),
    "Error Code: 105": ("Failed", "Insufficient space in disk library mount path"),
    "Error Code: 106": ("Failed", "VMware snapshot quiesce timeout"),
    "Error Code: 107": ("Failed", "Authentication to database instance failed"),
    "Error Code: 108": ("Completed w/ errors", "One or more files skipped: access denied"),
    "Error Code: 109": ("Completed w/ errors", "One or more files skipped: file in use"),
    "Error Code: 110": ("Completed w/ errors", "Some items could not be read from source"),
}

# Which failure codes make sense for which agent type
AGENT_FAIL_CODES = {
    "VMware":              ["Error Code: 101", "Error Code: 105", "Error Code: 106"],
    "SQL Server":          ["Error Code: 101", "Error Code: 104", "Error Code: 107"],
    "Oracle":              ["Error Code: 101", "Error Code: 104", "Error Code: 107"],
    "Windows File System": ["Error Code: 101", "Error Code: 103", "Error Code: 105"],
    "Linux File System":   ["Error Code: 101", "Error Code: 105"],
    "SAP HANA":            ["Error Code: 101", "Error Code: 104", "Error Code: 105"],
}
WARN_CODES = ["Error Code: 108", "Error Code: 109", "Error Code: 110"]
MEDIA_AGENT_DOWN_CODE = "Error Code: 102"

# How many of each anomaly to plant (13 total)
ANOMALY_PLAN = {
    "repeat_failure": 2,
    "duration_spike": 3,
    "size_drop": 3,
    "size_spike": 2,
    "missed_backup": 2,
    "media_agent_outage": 1,
}

NOISE_FAIL_RATE = 0.02
NOISE_WARN_RATE = 0.03
BASELINE_NIGHTS = 4      # no anomalies in the first N nights so baselines exist
WINDOW_START_HOUR = 20   # backup window opens 8 PM


# ---------------------------------------------------------------------------
# Building the environment and normal jobs
# ---------------------------------------------------------------------------

def build_clients(rng, n_clients):
    """Create clients spread evenly across sites and media agents."""
    agent_types = list(AGENT_PROFILES)
    weights = [AGENT_PROFILES[a][4] for a in agent_types]
    site_names = list(SITES)
    clients = []
    counters = {}
    for i in range(n_clients):
        site = site_names[i % len(site_names)]
        media_agent = SITES[site]["media_agents"][(i // len(site_names)) % 2]
        agent = rng.choices(agent_types, weights=weights)[0]
        prefix, size_rng, tput_rng, change_rng, _ = AGENT_PROFILES[agent]
        key = (prefix, site)
        counters[key] = counters.get(key, 0) + 1
        clients.append({
            "client": f"{prefix}-{SITES[site]['code']}-{counters[key]:03d}",
            "agent_type": agent,
            "site": site,
            "media_agent": media_agent,
            "base_full_gb": rng.uniform(*size_rng),
            "throughput_gbph": rng.uniform(*tput_rng),
            "change_rate": rng.uniform(*change_rng),
            "full_weekday": rng.randrange(7),      # weekly full on this day
            "start_offset_min": rng.randrange(0, 330),
        })
    return clients


def duration_for(size_gb, throughput_gbph, rng):
    return size_gb / throughput_gbph * 60 * rng.uniform(0.85, 1.2) + rng.uniform(3, 10)


def build_normal_jobs(rng, clients, nights):
    """Return {(client_index, night_index): job dict} with every job Completed."""
    jobs = {}
    for ci, c in enumerate(clients):
        for ni, night in enumerate(nights):
            level = "Full" if night.weekday() == c["full_weekday"] else "Incremental"
            if level == "Full":
                size = c["base_full_gb"] * rng.uniform(0.97, 1.03)
            else:
                size = c["base_full_gb"] * c["change_rate"] * rng.uniform(0.7, 1.3)
            start = (datetime.combine(night, datetime.min.time())
                     + timedelta(hours=WINDOW_START_HOUR,
                                 minutes=c["start_offset_min"] + rng.randint(-10, 10)))
            jobs[(ci, ni)] = {
                "client": c["client"],
                "agent_type": c["agent_type"],
                "site": c["site"],
                "media_agent": c["media_agent"],
                "backup_level": level,
                "start": start,
                "duration_min": duration_for(size, c["throughput_gbph"], rng),
                "size_gb": size,
                "status": "Completed",
                "error_code": "",
            }
    return jobs


def make_failed(job, code, rng):
    job["status"] = "Failed"
    job["error_code"] = code
    job["size_gb"] *= rng.uniform(0.0, 0.3)
    job["duration_min"] = max(1.0, job["duration_min"] * rng.uniform(0.05, 0.4))


# ---------------------------------------------------------------------------
# Planting anomalies
# ---------------------------------------------------------------------------

def plant_anomalies(rng, clients, nights, jobs):
    """Modify jobs in place to plant anomalies. Returns (anomalies, reserved, no_noise).

    Each anomaly records the (client_index, night_index) cells it touched, so
    job IDs can be resolved after IDs are assigned.
    """
    n_clients, n_nights = len(clients), len(nights)
    reserved = set()     # cells owned by an anomaly
    no_noise = set()     # cells where background noise must not land
    anomaly_clients = set()
    anomalies = []

    def free(ci, ni):
        return (ci, ni) not in reserved

    def pick_cell(condition=lambda ci, ni: True):
        for _ in range(10_000):
            ci = rng.randrange(n_clients)
            ni = rng.randrange(BASELINE_NIGHTS, n_nights)
            if ci in anomaly_clients:
                continue
            if free(ci, ni) and condition(ci, ni):
                return ci, ni
        raise RuntimeError("Could not place anomaly; try more clients or nights")

    def record(kind, cells, description, **extra):
        reserved.update(cells)
        anomalies.append({"type": kind, "cells": cells,
                          "description": description, **extra})

    # 1. Media agent outage: every job on one MA fails on the same night
    for _ in range(ANOMALY_PLAN["media_agent_outage"]):
        site = rng.choice(list(SITES))
        ma = rng.choice(SITES[site]["media_agents"])
        ni = rng.randrange(BASELINE_NIGHTS, n_nights)
        cells = [(ci, ni) for ci, c in enumerate(clients) if c["media_agent"] == ma]
        for cell in cells:
            make_failed(jobs[cell], MEDIA_AGENT_DOWN_CODE, rng)
        record("media_agent_outage", cells,
               f"Media agent {ma} offline; all {len(cells)} jobs on it failed the same "
               f"night. Should be reported as ONE root cause, not {len(cells)} failures.",
               media_agent=ma, site=site)

    # 2. Repeat failures: same client fails 3-4 consecutive nights, same error
    for _ in range(ANOMALY_PLAN["repeat_failure"]):
        streak = rng.choice([3, 4])
        for _try in range(10_000):
            ci = rng.randrange(n_clients)
            start = rng.randrange(BASELINE_NIGHTS, n_nights - streak + 1)
            cells = [(ci, ni) for ni in range(start, start + streak)]
            if ci not in anomaly_clients and all(free(*cell) for cell in cells):
                break
        else:
            raise RuntimeError("Could not place repeat failure")
        code = rng.choice(AGENT_FAIL_CODES[clients[ci]["agent_type"]])
        for cell in cells:
            make_failed(jobs[cell], code, rng)
        anomaly_clients.add(ci)
        no_noise.update({(ci, start - 1), (ci, start + streak)})
        record("repeat_failure", cells,
               f"Failed {streak} nights in a row with {code}; an ongoing issue, not a blip.",
               error_code=code)

    # Size/duration anomalies go on incrementals only: a 14-night report has just
    # two fulls per client, too few to establish a "normal" to compare against.
    def is_incr(ci, ni):
        return jobs[(ci, ni)]["backup_level"] == "Incremental"

    # 3. Duration spikes: completes, but runs 4-6x longer than normal
    for _ in range(ANOMALY_PLAN["duration_spike"]):
        ci, ni = pick_cell(is_incr)
        factor = rng.uniform(4, 6)
        jobs[(ci, ni)]["duration_min"] *= factor
        anomaly_clients.add(ci)
        record("duration_spike", [(ci, ni)],
               f"Completed, but ran ~{factor:.1f}x longer than this client's normal "
               f"{jobs[(ci, ni)]['backup_level'].lower()} duration.")

    # 4. Size drops: "Completed" but ~90% less data than normal (silent failure)
    for _ in range(ANOMALY_PLAN["size_drop"]):
        ci, ni = pick_cell(is_incr)
        factor = rng.uniform(0.05, 0.15)
        job = jobs[(ci, ni)]
        job["size_gb"] *= factor
        job["duration_min"] = duration_for(job["size_gb"], clients[ci]["throughput_gbph"], rng)
        anomaly_clients.add(ci)
        record("size_drop", [(ci, ni)],
               f"Status Completed, but backed up only ~{factor:.0%} of normal "
               f"{job['backup_level'].lower()} size. Silent failure; status filters miss it.")

    # 5. Size spikes: incremental 5-10x normal (abnormal change rate)
    for _ in range(ANOMALY_PLAN["size_spike"]):
        ci, ni = pick_cell(is_incr)
        factor = rng.uniform(5, 10)
        job = jobs[(ci, ni)]
        job["size_gb"] *= factor
        job["duration_min"] = duration_for(job["size_gb"], clients[ci]["throughput_gbph"], rng)
        anomaly_clients.add(ci)
        record("size_spike", [(ci, ni)],
               f"Incremental ~{factor:.1f}x normal size. Abnormal change rate; "
               f"possible ransomware encryption or mass data change.")

    # 6. Missed backups: the job never ran (row removed later)
    for _ in range(ANOMALY_PLAN["missed_backup"]):
        ci, ni = pick_cell()
        jobs[(ci, ni)]["missed"] = True
        anomaly_clients.add(ci)
        record("missed_backup", [(ci, ni)],
               "No job ran for this client on this night. Failure-only reports never show this.")

    return anomalies, reserved, no_noise


def add_background_noise(rng, clients, nights, jobs, reserved, no_noise):
    """One-off failures and warnings that are NOT anomalies."""
    noise = {"one_off_failures": [], "completed_with_errors": []}
    for ci in range(len(clients)):
        prev_failed = False
        for ni in range(len(nights)):
            cell = (ci, ni)
            job = jobs[cell]
            if cell in reserved or cell in no_noise:
                prev_failed = job["status"] == "Failed"
                continue
            r = rng.random()
            if r < NOISE_FAIL_RATE and not prev_failed:
                make_failed(job, rng.choice(AGENT_FAIL_CODES[job["agent_type"]]), rng)
                noise["one_off_failures"].append(cell)
            elif r < NOISE_FAIL_RATE + NOISE_WARN_RATE:
                job["status"] = "Completed w/ errors"
                job["error_code"] = rng.choice(WARN_CODES)
                job["size_gb"] *= rng.uniform(0.95, 0.99)
                noise["completed_with_errors"].append(cell)
            prev_failed = job["status"] == "Failed"
    return noise


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nights", type=int, default=14)
    p.add_argument("--clients", type=int, default=60)
    p.add_argument("--end-date", default="2026-10-11",
                   help="Last backup night (YYYY-MM-DD). Fixed so seeds are reproducible.")
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()

    rng = random.Random(args.seed)
    end = date.fromisoformat(args.end_date)
    nights = [end - timedelta(days=args.nights - 1 - i) for i in range(args.nights)]

    clients = build_clients(rng, args.clients)
    jobs = build_normal_jobs(rng, clients, nights)
    anomalies, reserved, no_noise = plant_anomalies(rng, clients, nights, jobs)
    noise = add_background_noise(rng, clients, nights, jobs, reserved, no_noise)

    # Drop missed backups, sort by start time, assign job IDs
    kept = sorted(((cell, j) for cell, j in jobs.items() if not j.get("missed")),
                  key=lambda x: (x[1]["start"], x[1]["client"]))
    job_id_for = {}
    rows = []
    for i, (cell, j) in enumerate(kept):
        job_id = 4_810_000 + i
        job_id_for[cell] = job_id
        end_time = j["start"] + timedelta(minutes=j["duration_min"])
        rows.append({
            "job_id": job_id,
            "client": j["client"],
            "agent_type": j["agent_type"],
            "site": j["site"],
            "media_agent": j["media_agent"],
            "backup_level": j["backup_level"],
            "start_time": j["start"].strftime("%Y-%m-%d %H:%M"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M"),
            "duration_min": round(j["duration_min"]),
            "size_gb": round(j["size_gb"], 1),
            "status": j["status"],
            "error_code": j["error_code"],
            "error_description": ERROR_CATALOG[j["error_code"]][1] if j["error_code"] else "",
        })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "backup_jobs.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Answer key: resolve cells into client names, night dates and job IDs
    key_anomalies = []
    for idx, a in enumerate(anomalies, start=1):
        cells = a.pop("cells")
        clients_hit = sorted({clients[ci]["client"] for ci, _ in cells})
        key_anomalies.append({
            "id": f"A{idx:02d}",
            "type": a.pop("type"),
            "clients": clients_hit,
            "nights": sorted({nights[ni].isoformat() for _, ni in cells}),
            "job_ids": sorted(job_id_for[c] for c in cells if c in job_id_for),
            **a,
        })

    answer_key = {
        "seed": args.seed,
        "nights": [nights[0].isoformat(), nights[-1].isoformat()],
        "total_clients": len(clients),
        "total_jobs": len(rows),
        "anomaly_count": len(key_anomalies),
        "anomalies": key_anomalies,
        "background_noise_not_anomalies": {
            "note": "One-off events a good detector should NOT escalate as anomalies.",
            "one_off_failure_job_ids": sorted(job_id_for[c] for c in noise["one_off_failures"]),
            "completed_with_errors_job_ids": sorted(job_id_for[c] for c in noise["completed_with_errors"]),
        },
    }
    key_path = out / "answer_key.json"
    key_path.write_text(json.dumps(answer_key, indent=2))

    # Console summary
    print(f"Wrote {len(rows)} jobs ({len(clients)} clients x {len(nights)} nights, "
          f"minus missed) to {csv_path}")
    print(f"Wrote answer key with {len(key_anomalies)} planted anomalies to {key_path}")
    print(f"Background noise: {len(noise['one_off_failures'])} one-off failures, "
          f"{len(noise['completed_with_errors'])} completed w/ errors")
    for a in key_anomalies:
        print(f"  {a['id']}  {a['type']:<20} {', '.join(a['clients'][:2])}"
              f"{' ...' if len(a['clients']) > 2 else ''}  {', '.join(a['nights'])}")


if __name__ == "__main__":
    main()
