#!/usr/bin/env python3
"""
evaluate.py

Scores the detector's findings against the generator's answer key.

This is the step that turns "I built a detector" into "my detector caught
13 of 13 planted anomalies with 0 false alarms." The detector never reads the
answer key; only this script does.

Scoring rules:
    - A planted anomaly counts as CAUGHT if some finding has the same type and
      covers it: shares a job ID, or (for missed backups, which have no job)
      names the same client and night.
    - A finding that matches no planted anomaly is a FALSE ALARM.
    - Background noise (one-off failures and warnings) should only appear in the
      informational one-off list. If a noise job shows up inside a real finding,
      that's counted as noise escalated.

Usage:
    python evaluate.py findings.json answer_key.json
"""

import argparse
import json
import sys


def matches(finding, anomaly):
    if finding["type"] != anomaly["type"]:
        return False
    if anomaly["job_ids"]:
        return bool(set(finding["job_ids"]) & set(anomaly["job_ids"]))
    return (set(finding["clients"]) & set(anomaly["clients"])
            and set(finding["nights"]) & set(anomaly["nights"]))


def evaluate(findings_doc, key):
    findings = findings_doc["findings"]
    anomalies = key["anomalies"]

    caught, missed = [], []
    matched_finding_ids = set()
    for a in anomalies:
        hits = [f for f in findings if matches(f, a)]
        if hits:
            caught.append((a, hits))
            matched_finding_ids.update(f["id"] for f in hits)
        else:
            missed.append(a)

    false_alarms = [f for f in findings if f["id"] not in matched_finding_ids]

    noise = key["background_noise_not_anomalies"]
    noise_ids = set(noise["one_off_failure_job_ids"]) | set(noise["completed_with_errors_job_ids"])
    escalated_noise = sorted(noise_ids & {j for f in findings for j in f["job_ids"]})

    n = len(anomalies)
    return {
        "planted_anomalies": n,
        "caught": len(caught),
        "missed": len(missed),
        "false_alarms": len(false_alarms),
        "recall": round(len(caught) / n, 3) if n else None,
        "precision": round((len(findings) - len(false_alarms)) / len(findings), 3) if findings else None,
        "noise_jobs": len(noise_ids),
        "noise_escalated": len(escalated_noise),
        "by_anomaly": [
            {"id": a["id"], "type": a["type"], "caught": bool(h),
             "finding_ids": [f["id"] for f in h]}
            for a, h in caught
        ] + [{"id": a["id"], "type": a["type"], "caught": False, "finding_ids": []} for a in missed],
        "missed_detail": missed,
        "false_alarm_detail": false_alarms,
        "escalated_noise_job_ids": escalated_noise,
    }


def main():
    p = argparse.ArgumentParser(description="Score findings against the answer key.")
    p.add_argument("findings")
    p.add_argument("answer_key")
    p.add_argument("--out", help="Write the full results JSON to this path")
    args = p.parse_args()

    with open(args.findings) as f:
        findings_doc = json.load(f)
    with open(args.answer_key) as f:
        key = json.load(f)

    r = evaluate(findings_doc, key)

    print(f"Caught {r['caught']} of {r['planted_anomalies']} planted anomalies "
          f"(recall {r['recall']:.0%})")
    print(f"False alarms: {r['false_alarms']} (precision "
          f"{r['precision']:.0%})" if r["precision"] is not None else "No findings")
    print(f"Background noise escalated: {r['noise_escalated']} of {r['noise_jobs']} noise jobs\n")

    for row in sorted(r["by_anomaly"], key=lambda x: x["id"]):
        mark = "CAUGHT" if row["caught"] else "MISSED"
        print(f"  {row['id']}  {row['type']:<20} {mark}  {', '.join(row['finding_ids'])}")
    for f in r["false_alarm_detail"]:
        print(f"  FALSE ALARM  {f['id']}  {f['type']}: {f['summary']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nWrote {args.out}")

    # Non-zero exit if anything was missed or falsely flagged; handy for automated checks later
    sys.exit(0 if r["missed"] == 0 and r["false_alarms"] == 0 else 1)


if __name__ == "__main__":
    main()
