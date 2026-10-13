#!/usr/bin/env python3
"""
summarizer.py

Turns the detector's findings into a plain-English morning summary with an LLM.

Division of labor (the core design decision of this project):
    - CODE decides what is wrong: detection, counts, severity, overall status.
    - The LLM decides how to EXPLAIN it: wording, grouping, priorities, next steps.
    The LLM never sees the raw CSV, only the structured findings. It can't
    miscount 838 rows if it never has to count them.

Guardrails:
    1. Structured output. The API is asked for JSON that matches a fixed schema
       (the "structured outputs" feature), so the app always gets the same fields.
    2. ID check. Every priority must cite finding IDs (F01, F02...). Unknown IDs
       are flagged, and so is any critical/high finding the summary left out.
    3. Number grounding check. Every number in the summary is compared with the
       numbers in the findings. A number that doesn't trace back is flagged as a
       possible hallucination.
    4. Fallback. With no API key, or if the API call fails, a code-only summary
       is produced, so the app always works.

Usage:
    python detector.py backup_jobs.csv --out findings.json
    python summarizer.py findings.json
    python summarizer.py findings.json --no-llm        # code-only summary
"""

import argparse
import json
import os
import re

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = 8000  # a ceiling, not a target: you only pay for tokens actually generated

SYSTEM_PROMPT = """You are the lead engineer on a backup operations team, writing the 7 a.m. \
handover for the on-call engineer who is starting their shift. They are busy and will read \
this on a phone. Be direct and specific.

You will receive structured findings that code already detected in last night's backup job \
report. The findings are correct; your job is to explain and prioritize them, not to find new ones.

Rules:
- Use ONLY facts in the findings. Never invent clients, numbers, dates, error codes or causes.
- Copy numbers exactly as they appear. Do not do new arithmetic.
- Every priority must cite the finding IDs it covers (for example F01).
- Include every critical and high finding in a priority. Group findings that share a cause \
or a place (same media agent, same site, same night) into one priority.
- Order priorities by urgency: anything still failing or affecting many clients first.
- Next steps are checks an engineer can do now. Phrase causes as possibilities to verify, \
never as confirmed facts. Do not quote vendor-specific commands or error code meanings that \
are not in the findings.
- Medium findings may be summarized briefly in also_noted instead of getting their own priority.
- One-off issues are informational: mention how many there were and that they only need \
attention if they repeat.
- Be concise. Keep what_happened and why_it_matters together under about 60 words per priority, \
and give at most 2 next steps.
- Plain text only inside every field: no Markdown, no asterisks, no backticks, no bold or headings.
- Plain English. No hype, no emojis."""

# The shape the model must answer in. The API enforces this schema.
# Structured outputs require additionalProperties: false on every object, and don't
# allow keywords like minLength/maxItems, so limits live in the descriptions instead.
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": "One sentence, max 30 words: the most important thing to know this morning.",
        },
        "priorities": {
            "type": "array",
            "description": "Ordered most urgent first. Covers every critical and high finding.",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short label, max 10 words."},
                    "finding_ids": {"type": "array", "items": {"type": "string"}},
                    "what_happened": {"type": "string", "description": "1-2 sentences, facts only."},
                    "why_it_matters": {"type": "string", "description": "1 sentence: the risk if ignored."},
                    "next_steps": {
                        "type": "array", "items": {"type": "string"},
                        "description": "1-3 concrete checks to do now.",
                    },
                },
                "required": ["title", "finding_ids", "what_happened", "why_it_matters", "next_steps"],
                "additionalProperties": False,
            },
        },
        "also_noted": {
            "type": "string",
            "description": "1-3 sentences on medium findings not covered above. Empty string if none.",
        },
        "one_offs_note": {
            "type": "string",
            "description": "1 sentence on the informational one-off failures and warnings.",
        },
    },
    "required": ["headline", "priorities", "also_noted", "one_offs_note"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Decisions made in code, not by the LLM
# ---------------------------------------------------------------------------

def overall_status(result):
    """Red / amber / green from severities. Deterministic, so it's never left to the model."""
    severities = {f["severity"] for f in result["findings"]}
    if "critical" in severities:
        return "red"
    if "high" in severities:
        return "amber"
    return "green"


def build_llm_input(result):
    """The compact, structured view of the night that the LLM is allowed to see."""
    one_offs = result["one_off_issues"]
    by_status = {}
    for o in one_offs:
        by_status[o["status"]] = by_status.get(o["status"], 0) + 1
    return {
        "report": result["report"],
        "overall_status": overall_status(result),
        "findings": [{
            "id": f["id"],
            "type": f["type"],
            "severity": f["severity"],
            "clients": f["clients"],
            "media_agent": f.get("media_agent"),
            "site": f.get("site"),
            "nights": f["nights"],
            "evidence": f["evidence"],
            "summary": f["summary"],
        } for f in result["findings"]],
        "one_off_issues": {"count": len(one_offs), "by_status": by_status},
    }


# ---------------------------------------------------------------------------
# Guardrail checks on the LLM's answer
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _numbers_in(text):
    return {m.group() for m in _NUMBER.finditer(text)}


def _summary_text(summary):
    parts = [summary.get("headline", ""), summary.get("also_noted", ""), summary.get("one_offs_note", "")]
    for p in summary.get("priorities", []):
        parts += [p.get("title", ""), p.get("what_happened", ""), p.get("why_it_matters", "")]
        parts += p.get("next_steps", [])
    return "\n".join(parts)


def validate_summary(summary, result):
    """Return a list of human-readable warnings. Empty list = passed every check."""
    warnings = []
    known = {f["id"] for f in result["findings"]}
    cited = {fid for p in summary.get("priorities", []) for fid in p.get("finding_ids", [])}

    unknown = sorted(cited - known)
    if unknown:
        warnings.append(f"Cites finding IDs that don't exist: {', '.join(unknown)}")

    must_cover = {f["id"] for f in result["findings"] if f["severity"] in ("critical", "high")}
    uncovered = sorted(must_cover - cited)
    if uncovered:
        warnings.append(f"Critical/high findings left out of the priorities: {', '.join(uncovered)}")

    # Number grounding: every number in the summary must exist in the input.
    # Rounded forms count too (238.1 GB may be written as 238 GB).
    source_numbers = _numbers_in(json.dumps(build_llm_input(result)))
    allowed = set(source_numbers)
    for n in source_numbers:
        value = float(n)
        allowed.update({str(round(value)), f"{value:.1f}", str(int(value))})
        if value <= 1:  # ratios like 0.09 may be written as 9%
            allowed.add(str(round(value * 100)))
    allowed.update({str(i) for i in range(0, 11)})  # small counts ("3 checks") are harmless
    ungrounded = sorted(_numbers_in(_summary_text(summary)) - allowed, key=float)
    if ungrounded:
        warnings.append("Numbers not found in the findings (possible hallucination): "
                        + ", ".join(ungrounded))
    return warnings


_MARKDOWN_CHARS = re.compile(r"\*+|`+")


def clean_text(value):
    """Strip Markdown emphasis characters from every string, even if the model adds them.
    The UI does its own formatting; stray asterisks only add clutter or accidental italics."""
    if isinstance(value, str):
        return _MARKDOWN_CHARS.sub("", value).strip()
    if isinstance(value, list):
        return [clean_text(v) for v in value]
    if isinstance(value, dict):
        return {k: clean_text(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def code_only_summary(result, reason=""):
    """Deterministic summary used when no LLM is available. No API, no cost."""
    findings = result["findings"]
    r = result["report"]
    top = [f for f in findings if f["severity"] in ("critical", "high")]
    if findings:
        headline = (f"{len(findings)} findings for the night of {r['last_night']}: "
                    f"{sum(f['severity'] == 'critical' for f in findings)} critical, "
                    f"{sum(f['severity'] == 'high' for f in findings)} high, "
                    f"{sum(f['severity'] == 'medium' for f in findings)} medium.")
    else:
        headline = f"No anomalies detected across {r['total_jobs']} jobs."
    medium = [f for f in findings if f["severity"] == "medium"]
    return {
        "headline": headline,
        "priorities": [{
            "title": f"{f['type'].replace('_', ' ').capitalize()}: {', '.join(f['clients'][:3])}"
                     + (" ..." if len(f["clients"]) > 3 else ""),
            "finding_ids": [f["id"]],
            "what_happened": f["summary"],
            "why_it_matters": "",
            "next_steps": [],
        } for f in top],
        "also_noted": " ".join(f["summary"] for f in medium),
        "one_offs_note": (f"{len(result['one_off_issues'])} one-off failures or warnings; "
                          f"informational unless they repeat."),
        "_meta": {"source": "code", "reason": reason},
    }


def _make_client(api_key):
    import anthropic  # imported here so the code-only path works without the SDK
    return anthropic.Anthropic(api_key=api_key)


def llm_summary(result, api_key=None, model=DEFAULT_MODEL, client=None):
    """Ask the LLM for the summary. Falls back to the code-only summary on any failure."""
    if client is None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return code_only_summary(result, "No API key configured")
        client = _make_client(key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": SUMMARY_SCHEMA}},
            messages=[{
                "role": "user",
                "content": ("Here are last night's findings as JSON. Write the morning summary.\n\n"
                            + json.dumps(build_llm_input(result), indent=1)),
            }],
        )
    except Exception as exc:  # network, auth, rate limit, bad model name...
        return code_only_summary(result, f"LLM call failed: {type(exc).__name__}: {exc}")

    stop = getattr(response, "stop_reason", None)
    if stop in ("refusal", "max_tokens"):  # output may be cut off or off-schema
        used = getattr(getattr(response, "usage", None), "output_tokens", "?")
        return code_only_summary(result, f"LLM stopped early ({stop} after {used} output tokens)")
    text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
    try:
        summary = json.loads(text)
    except (TypeError, ValueError):
        return code_only_summary(result, "LLM did not return valid JSON")
    if not isinstance(summary, dict) or not set(SUMMARY_SCHEMA["required"]) <= set(summary):
        return code_only_summary(result, "LLM response was missing required fields")

    summary = clean_text(dict(summary))
    summary["_meta"] = {
        "source": "llm",
        "model": model,
        "input_tokens": getattr(response.usage, "input_tokens", None),
        "output_tokens": getattr(response.usage, "output_tokens", None),
        "warnings": validate_summary(summary, result),
    }
    return summary


def to_text(summary, result):
    """Render a summary as plain text for the terminal (no Markdown symbols)."""
    lines = [f"OVERALL: {overall_status(result).upper()}", summary["headline"], ""]
    for i, p in enumerate(summary["priorities"], start=1):
        lines.append(f"{i}. {p['title']} ({', '.join(p['finding_ids'])})")
        lines.append(f"   {p['what_happened']}")
        if p.get("why_it_matters"):
            lines.append(f"   Why it matters: {p['why_it_matters']}")
        for step in p.get("next_steps", []):
            lines.append(f"   - {step}")
        lines.append("")
    if summary.get("also_noted"):
        lines += [f"Also noted: {summary['also_noted']}", ""]
    if summary.get("one_offs_note"):
        lines.append(summary["one_offs_note"])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Write the morning summary from findings JSON.")
    p.add_argument("findings")
    p.add_argument("--no-llm", action="store_true", help="Use the code-only summary")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    with open(args.findings) as f:
        result = json.load(f)

    summary = code_only_summary(result, "--no-llm") if args.no_llm else llm_summary(result, model=args.model)
    print(to_text(summary, result))
    meta = summary["_meta"]
    print(f"\n---\nSource: {meta['source']}"
          + (f" ({meta.get('model')}, {meta.get('input_tokens')} in / {meta.get('output_tokens')} out tokens)"
             if meta["source"] == "llm" else f" ({meta.get('reason')})"))
    for w in meta.get("warnings", []):
        print(f"WARNING: {w}")
    if meta["source"] == "llm" and not meta.get("warnings"):
        print("Checks passed: all finding IDs valid, all critical/high covered, all numbers grounded.")


if __name__ == "__main__":
    main()
