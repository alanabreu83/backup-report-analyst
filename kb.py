#!/usr/bin/env python3
"""
kb.py: the error-code and procedure knowledge base.

The knowledge base (KB) is a JSON file YOU write from your own experience and
public vendor documentation. The LLM is only allowed to take next steps from
entries retrieved from it; if nothing matches, it must say so.

Two kinds of entry:
    error_code  - what one error code means and what to check (match by code)
    procedure   - what to do for one kind of anomaly, e.g. a silent size drop
                  (match by finding type)

Commands:
    python kb.py scaffold backup_jobs.csv     create knowledge_base.json with stubs to fill in
    python kb.py scaffold backup_jobs.csv --merge   add stubs for codes that are not there yet
    python kb.py lint                         check the file; exit code 1 if it has errors
    python kb.py search "snapshot failed"     try retrieval from the terminal
    python kb.py search --code "Error Code: 101"

Never put Travelers (or any employer's) internal documents, ticket text, hostnames or
runbooks in this file. Use your own general knowledge and public vendor docs only.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_KB_PATH = Path(__file__).parent / "knowledge_base.json"

FINDING_TYPES = ["media_agent_outage", "repeat_failure", "missed_backup",
                 "size_drop", "size_spike", "duration_spike"]
KINDS = ("error_code", "procedure")
LIST_FIELDS = ["error_codes", "finding_types", "likely_causes", "checks", "fixes", "tags"]
TEXT_FIELDS = ["id", "kind", "title", "symptoms", "source"]
PLACEHOLDER = "TODO"
UNREVIEWED_MARKER = "STARTER DRAFT"
MAX_ENTRY_CHARS = 1500  # keeps the LLM input small and each entry focused


class KBError(ValueError):
    """The KB file can't be read or isn't shaped like a KB."""


def load_kb(path=DEFAULT_KB_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except FileNotFoundError:
        raise KBError(f"Knowledge base not found: {path}") from None
    except ValueError as exc:
        raise KBError(f"{path} is not valid JSON: {exc}") from exc
    entries = doc.get("entries") if isinstance(doc, dict) else None
    if not isinstance(entries, list):
        raise KBError(f'{path} must look like {{"entries": [ ... ]}}')
    return [normalize_entry(e) for e in entries if isinstance(e, dict)]


def normalize_entry(e):
    """Fill missing fields so the rest of the code can rely on every key existing."""
    out = {k: str(e.get(k, "") or "").strip() for k in TEXT_FIELDS}
    for k in LIST_FIELDS:
        v = e.get(k, [])
        out[k] = [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []
    return out


def usable_entries(entries):
    """Entries safe to show the LLM: anything still holding TODO placeholders is left out."""
    return [e for e in entries if PLACEHOLDER not in json.dumps(e)]


def entry_text(e):
    """All the searchable text of an entry, as one string."""
    parts = [e["title"], e["title"], e["symptoms"], " ".join(e["error_codes"]),
             " ".join(t.replace("_", " ") for t in e["finding_types"]),
             " ".join(e["likely_causes"]), " ".join(e["checks"]),
             " ".join(e["fixes"]), " ".join(e["tags"])]
    return " ".join(p for p in parts if p)


def kb_digest(entries):
    """Short fingerprint of the KB content, used so cached summaries refresh when it changes."""
    blob = json.dumps(entries, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def lint(entries):
    """Return (errors, warnings). Errors block use; warnings are reminders."""
    from retriever import normalize_code  # local import: retriever imports this module
    errors, warnings = [], []
    seen_ids, seen_codes = set(), {}
    if not entries:
        errors.append("The KB has no entries.")
    for i, e in enumerate(entries, start=1):
        name = e["id"] or f"entry #{i}"
        if not e["id"]:
            errors.append(f"{name}: missing id")
        elif e["id"] in seen_ids:
            errors.append(f"{name}: duplicate id")
        seen_ids.add(e["id"])
        if e["kind"] not in KINDS:
            errors.append(f"{name}: kind must be one of {', '.join(KINDS)}")
        if not e["title"]:
            errors.append(f"{name}: missing title")
        if not e["checks"] and not e["fixes"]:
            errors.append(f"{name}: needs at least one check or fix")
        if not e["source"]:
            errors.append(f"{name}: missing source (your own experience, or a public doc URL)")
        if e["kind"] == "error_code" and not e["error_codes"]:
            errors.append(f"{name}: error_code entry has no error_codes")
        if e["kind"] == "procedure" and not e["finding_types"]:
            errors.append(f"{name}: procedure entry has no finding_types")
        for t in e["finding_types"]:
            if t not in FINDING_TYPES:
                errors.append(f"{name}: unknown finding type '{t}' (valid: {', '.join(FINDING_TYPES)})")
        for c in e["error_codes"]:
            n = normalize_code(c)
            if n in seen_codes and seen_codes[n] != e["id"]:
                errors.append(f"{name}: code '{c}' is already used by {seen_codes[n]}")
            seen_codes[n] = e["id"]
        blob = json.dumps(e)
        if PLACEHOLDER in blob:
            errors.append(f"{name}: still contains {PLACEHOLDER} placeholders")
        if UNREVIEWED_MARKER in e["source"]:
            warnings.append(f"{name}: source says '{UNREVIEWED_MARKER}'; rewrite from your experience, then change the source")
        if len(entry_text(e)) > MAX_ENTRY_CHARS:
            warnings.append(f"{name}: over {MAX_ENTRY_CHARS} characters; split it into two entries")
    return errors, warnings


# ---------------------------------------------------------------------------
# Scaffold
# ---------------------------------------------------------------------------

def scaffold(csv_path, out_path=DEFAULT_KB_PATH, merge=False):
    """Create stub entries for every error code in the report, plus one per finding type."""
    from retriever import normalize_code
    from schema import load_jobs
    df = load_jobs(csv_path)
    codes = (df[df["error_code"] != ""].groupby("error_code")["error_description"]
             .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "").sort_index())

    existing = []
    if merge and Path(out_path).exists():
        existing = load_kb(out_path)
    elif Path(out_path).exists():
        raise KBError(f"{out_path} already exists. Use --merge to add missing stubs, "
                      f"or delete it to start over.")

    have_codes = {normalize_code(c) for e in existing for c in e["error_codes"]}
    have_types = {t for e in existing if e["kind"] == "procedure" for t in e["finding_types"]}
    entries = list(existing)
    n = len(entries)

    for code, desc in codes.items():
        if normalize_code(code) in have_codes:
            continue
        n += 1
        entries.append({
            "id": f"KB-{n:03d}", "kind": "error_code", "title": f"{code}: {desc}".strip(": "),
            "error_codes": [code], "finding_types": [],
            "symptoms": f"{PLACEHOLDER}: how this shows up in the job report. Report text: {desc}",
            "likely_causes": [f"{PLACEHOLDER}: one cause per line, most common first"],
            "checks": [f"{PLACEHOLDER}: what an engineer can check right now"],
            "fixes": [f"{PLACEHOLDER}: the usual fix, or when to escalate"],
            "source": f"{PLACEHOLDER}: your own experience, or a public doc URL",
            "tags": [],
        })
    for t in FINDING_TYPES:
        if t in have_types:
            continue
        n += 1
        entries.append({
            "id": f"KB-{n:03d}", "kind": "procedure",
            "title": f"{PLACEHOLDER}: procedure for {t.replace('_', ' ')}",
            "error_codes": [], "finding_types": [t],
            "symptoms": f"{PLACEHOLDER}: what {t.replace('_', ' ')} looks like in the report",
            "likely_causes": [f"{PLACEHOLDER}: possible causes"],
            "checks": [f"{PLACEHOLDER}: what to check first"],
            "fixes": [f"{PLACEHOLDER}: what to do, or when to escalate"],
            "source": f"{PLACEHOLDER}: your own experience, or a public doc URL",
            "tags": [],
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, indent=2, ensure_ascii=False)
    return len(entries) - len(existing)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Manage the backup knowledge base.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scaffold", help="Create stub entries from a job report CSV")
    s.add_argument("csv")
    s.add_argument("--kb", default=str(DEFAULT_KB_PATH))
    s.add_argument("--merge", action="store_true")
    l = sub.add_parser("lint", help="Check the KB file")
    l.add_argument("--kb", default=str(DEFAULT_KB_PATH))
    q = sub.add_parser("search", help="Try retrieval")
    q.add_argument("query", nargs="?", default="")
    q.add_argument("--code", action="append", default=[])
    q.add_argument("--type", dest="finding_type")
    q.add_argument("--kb", default=str(DEFAULT_KB_PATH))
    q.add_argument("-k", type=int, default=3)
    args = p.parse_args()

    try:
        if args.cmd == "scaffold":
            added = scaffold(args.csv, args.kb, args.merge)
            print(f"Wrote {added} stub entries to {args.kb}. Replace every {PLACEHOLDER} "
                  f"with your own knowledge, then run: python kb.py lint")
        elif args.cmd == "lint":
            errors, warnings = lint(load_kb(args.kb))
            for w in warnings:
                print(f"warning: {w}")
            for e in errors:
                print(f"ERROR:   {e}")
            print(f"\n{len(errors)} errors, {len(warnings)} warnings")
            sys.exit(1 if errors else 0)
        elif args.cmd == "search":
            from retriever import Retriever
            hits = Retriever(usable_entries(load_kb(args.kb))).search(
                args.query, codes=args.code, finding_type=args.finding_type, k=args.k,
                extract_codes=True)
            if not hits:
                print("No matching entries.")
            for h in hits:
                print(f"{h['entry']['id']}  [{h['why']}, score {h['score']}]  {h['entry']['title']}")
    except KBError as exc:
        print(f"error: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
