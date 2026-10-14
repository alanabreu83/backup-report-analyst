"""
app.py: Backup Report Analyst (Streamlit UI)

Run locally:   streamlit run app.py
Pipeline:      CSV -> schema.load_jobs -> detector.detect -> summarizer.llm_summary -> this page

Public-demo safeguards (the deployed app is open to anyone with the link):
    - Summaries are cached by content, so repeat clicks on the sample report
      cost nothing after the first one.
    - Each browser session is capped at MAX_LLM_CALLS_PER_SESSION summaries.
    - Without an API key the app still works, using the code-only summary.
    - Set a monthly spend limit in the Anthropic Console as the final backstop.
"""

import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from detector import detect
from kb import DEFAULT_KB_PATH, KBError, kb_digest, load_kb, usable_entries
from retriever import Retriever, retrieve_for_result
from schema import SchemaError, load_jobs
from summarizer import DEFAULT_MODEL, code_only_summary, llm_summary, overall_status

SAMPLE_CSV = Path(__file__).parent / "backup_jobs.csv"
MAX_LLM_CALLS_PER_SESSION = 5
STATUS_LABEL = {"red": ":red[**RED**]", "amber": ":orange[**AMBER**]", "green": ":green[**GREEN**]"}
SEVERITY_LABEL = {"critical": ":red[critical]", "high": ":orange[high]", "medium": ":blue[medium]"}


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def get_api_key():
    """Local runs read .env; Streamlit Cloud reads the app's Secrets settings."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:  # no secrets file, or key not set
        return None


@st.cache_data(show_spinner=False)
def analyze(csv_bytes):
    """Validate and run the detector. Cached on file content."""
    return detect(load_jobs(io.BytesIO(csv_bytes)))


@st.cache_resource(show_spinner=False)
def load_retriever(path, mtime):
    """Load the knowledge base once per file version. Returns (retriever, digest, counts)."""
    all_entries = load_kb(path)
    usable = usable_entries(all_entries)
    return Retriever(usable), kb_digest(usable), (len(usable), len(all_entries))


class _DontCache(Exception):
    """Raised inside the cached function so a failed call is never cached."""
    def __init__(self, summary):
        super().__init__("LLM summary not cached")
        self.summary = summary


@st.cache_data(show_spinner=False)
def _cached_llm_summary(result_json, kb_json, model, _api_key):
    """Cached on the findings, the retrieved KB entries and the model, so the same report is
    only summarized once, and editing the KB refreshes it.
    (The leading underscore tells Streamlit not to use the key as part of the cache key.)"""
    summary = llm_summary(json.loads(result_json), api_key=_api_key, model=model,
                          kb_hits=json.loads(kb_json))
    if summary["_meta"]["source"] != "llm":
        raise _DontCache(summary)  # don't cache failures, so a retry can succeed
    return summary


def get_llm_summary(result, kb_hits, model, api_key):
    try:
        return _cached_llm_summary(json.dumps(result, sort_keys=True),
                                   json.dumps(kb_hits, sort_keys=True), model, api_key)
    except _DontCache as exc:
        return exc.summary


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Backup Report Analyst", layout="wide")
st.title("Backup Report Analyst")
st.caption("Code finds the problems. The LLM explains them. Every number traces back to the data.")

with st.sidebar:
    st.header("Report")
    source = st.radio("Data source", ["Sample report (synthetic)", "Upload a CSV"])
    uploaded = None
    if source == "Upload a CSV":
        uploaded = st.file_uploader("Job report CSV", type=["csv"])
        st.caption("Synthetic or anonymized data only. Uploaded data is sent to the LLM provider "
                   "when you generate a summary.")
    st.divider()
    st.caption("How it works")
    st.markdown(
        "1. **Validate** the report against a fixed schema\n"
        "2. **Detect** anomalies in code (pandas): outages, repeat failures, missed backups, "
        "silent size drops, size spikes, slow jobs\n"
        "3. **Explain** with an LLM that sees only the structured findings, never the raw rows\n"
        "4. **Look up** each finding's error codes and anomaly type in a knowledge base "
        "(exact code match + keyword search)\n"
        "5. **Check** the summary: valid finding IDs, full coverage, numbers grounded in the data, "
        "next steps traced to a knowledge base entry"
    )

if source == "Upload a CSV" and uploaded is None:
    st.info("Upload a job report CSV, or switch to the sample report in the sidebar.")
    st.stop()

csv_bytes = uploaded.getvalue() if uploaded is not None else SAMPLE_CSV.read_bytes()
report_digest = hashlib.sha256(csv_bytes).hexdigest()
try:
    result = analyze(csv_bytes)
except SchemaError as exc:
    st.error(f"This file doesn't match the expected report format: {exc}")
    st.stop()

report, findings = result["report"], result["findings"]

# --- Knowledge base lookup (plain code, no LLM) ---------------------------------
kb_hits, kb_note = {}, None
if Path(DEFAULT_KB_PATH).exists():
    try:
        retriever, kb_version, (n_usable, n_total) = load_retriever(
            str(DEFAULT_KB_PATH), Path(DEFAULT_KB_PATH).stat().st_mtime)
        if n_usable:
            kb_hits = retrieve_for_result(result, retriever)
        kb_note = f"Knowledge base: {n_usable} of {n_total} entries in use (version {kb_version})."
    except KBError as exc:
        st.sidebar.error(f"Knowledge base problem: {exc}")
else:
    kb_note = "No knowledge_base.json found, so next steps are not available."
with st.sidebar:
    if kb_note:
        st.caption(kb_note)
status = overall_status(result)

kb_lookup = {h["entry"]["id"]: h["entry"] for hits in kb_hits.values() for h in hits}


def kb_source_label(entry_id, lookup):
    """'KB-009 (title): source', with the source as a link when it is a web address."""
    e = lookup.get(entry_id)
    if not e:
        return entry_id
    src = e.get("source", "").strip()
    if src.lower().startswith(("http://", "https://")) and " " not in src:
        src = f"[{src.split('//', 1)[1].split('/', 1)[0]}]({src})"
    return f"{entry_id} ({e['title']}): {src}" if src else f"{entry_id} ({e['title']})"


# --- Headline numbers -------------------------------------------------------
st.markdown(f"#### Night of {report['last_night']} · Overall {STATUS_LABEL[status]}")
st.caption(f"Report covers {report['first_night']} to {report['last_night']} "
           f"({report['nights']} nights). Baselines come from each client's own history.")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Jobs", f"{report['total_jobs']:,}")
c2.metric("Clients", report["clients"])
c3.metric("Critical", sum(f["severity"] == "critical" for f in findings))
c4.metric("High", sum(f["severity"] == "high" for f in findings))
c5.metric("Medium", sum(f["severity"] == "medium" for f in findings))

# --- Morning summary --------------------------------------------------------
st.subheader("Morning summary")
api_key = get_api_key()
st.session_state.setdefault("llm_calls", 0)
st.session_state.setdefault("summary", None)
st.session_state.setdefault("summary_for", None)

col_a, col_b = st.columns([1, 3])
with col_a:
    clicked = st.button("Generate AI summary", type="primary", disabled=api_key is None)
with col_b:
    if api_key is None:
        st.caption("No API key configured, so the code-only summary is shown below.")
    else:
        st.caption(f"Model: {DEFAULT_MODEL}. Summaries are cached; "
                   f"limit {MAX_LLM_CALLS_PER_SESSION} per session.")

if clicked:
    if st.session_state["llm_calls"] >= MAX_LLM_CALLS_PER_SESSION:
        st.warning("Summary limit reached for this session. Showing the code-only summary.")
    else:
        st.session_state["llm_calls"] += 1
        with st.spinner("Writing the morning summary..."):
            st.session_state["summary"] = get_llm_summary(result, kb_hits, DEFAULT_MODEL, api_key)
            st.session_state["summary_for"] = report_digest

# Only show an AI summary if it was written for exactly this file
if st.session_state["summary_for"] == report_digest and st.session_state["summary"]:
    summary = st.session_state["summary"]
else:
    summary = code_only_summary(result, "Click Generate AI summary for the LLM version", kb_hits)
meta = summary["_meta"]

with st.container(border=True):
    st.markdown(f"**{summary['headline']}**")
    for i, p in enumerate(summary["priorities"], start=1):
        st.markdown(f"**{i}. {p['title']}** · `{', '.join(p['finding_ids'])}`")
        st.markdown(p["what_happened"])
        if p.get("why_it_matters"):
            st.markdown(f"*Why it matters:* {p['why_it_matters']}")
        if p.get("next_steps"):
            st.markdown("\n".join(f"- {s}" for s in p["next_steps"]))
        if p.get("kb_ids"):
            st.caption("Sources: " + " · ".join(kb_source_label(i, kb_lookup) for i in p["kb_ids"]))
    if summary.get("also_noted"):
        st.markdown(f"**Also noted:** {summary['also_noted']}")
    if summary.get("one_offs_note"):
        st.caption(summary["one_offs_note"])

if meta["source"] == "llm":
    tokens = f"{meta.get('input_tokens')} in / {meta.get('output_tokens')} out tokens"
    if meta["warnings"]:
        st.warning("Checks flagged this summary:\n\n" + "\n".join(f"- {w}" for w in meta["warnings"]))
    else:
        st.success(f"Checks passed: finding IDs valid, every critical/high finding covered, "
                   f"every number traced to the data. ({meta['model']}, {tokens})")
else:
    reason = meta.get("reason", "")
    if reason.startswith("LLM"):
        st.error(f"{reason}. Showing the code-only summary instead.")
    else:
        st.caption(f"Code-only summary. {reason}")

# --- Findings detail --------------------------------------------------------
st.subheader(f"Findings ({len(findings)})")
if findings:
    table = pd.DataFrame([{
        "ID": f["id"],
        "Severity": f["severity"],
        "Type": f["type"].replace("_", " "),
        "Clients": ", ".join(f["clients"][:3]) + (f" +{len(f['clients']) - 3} more" if len(f["clients"]) > 3 else ""),
        "Night(s)": ", ".join(f["nights"]),
        "Summary": f["summary"],
    } for f in findings])
    st.dataframe(table, hide_index=True)

    for f in findings:
        with st.expander(f"{f['id']} · {f['type'].replace('_', ' ')} · {', '.join(f['clients'][:2])}"
                         + (" ..." if len(f["clients"]) > 2 else "")):
            st.markdown(f"Severity: {SEVERITY_LABEL[f['severity']]}  \n{f['summary']}")
            hits = kb_hits.get(f["id"], [])
            if hits:
                st.markdown("**Knowledge base matches**")
                for h in hits:
                    st.markdown(f"- `{h['entry']['id']}` {h['entry']['title']} "
                                f"({h['why']}, score {h['score']})")
            elif kb_hits:
                st.markdown("**Knowledge base matches:** none")
            st.json({"evidence": f["evidence"], "job_ids": f["job_ids"],
                     "media_agent": f.get("media_agent"), "site": f.get("site")})
else:
    st.success("No anomalies detected.")

with st.expander(f"One-off issues ({len(result['one_off_issues'])}): informational, not escalated"):
    if result["one_off_issues"]:
        st.dataframe(pd.DataFrame(result["one_off_issues"]), hide_index=True)
    else:
        st.write("None.")

st.download_button("Download findings (JSON)", json.dumps(result, indent=2),
                   file_name="findings.json", mime="application/json")
