# Backup Report Analyst

**Every morning, someone on a backup team reads hundreds of overnight job results to find the few that
matter. This app does that first pass: code finds the problems, an LLM writes the handover in plain
English, and every next step cites a knowledge base entry.**

**[Live demo](https://backup-report-analyst.streamlit.app/)** · **[Video walkthrough](https://www.loom.com/share/07e868f1213c4962b15ae072e184bbfc)** · [Evaluation](EVALUATION.md) · Synthetic data only

![Morning summary](docs/screenshot.png)

## The problem

Backup reports are easy to read for failures and hard to read for everything else. A job marked
*Failed* is obvious. These are not:

- **A job that "Completed" but backed up 5% of its normal data.** Status says success. The data is not protected.
- **A client with no job at all.** Nothing failed, so a failure report never shows it.
- **Ten failures that are really one problem.** Every job on one media agent failed the same night. That is one outage, not ten tickets.
- **An incremental five times larger than normal.** It might be a data load, or it might be ransomware encrypting files.

I've spent 20 years running enterprise backup and recovery, and reading these reports is part of the job. This project is the tool I wanted.

## What it does

Given a nightly job report (838 jobs, 60 clients, 3 sites, 14 nights in the sample), the app:

1. **Validates** the report against a fixed schema, so a bad file fails loudly instead of producing a wrong answer.
2. **Detects** six kinds of anomaly in code: media agent outages, repeat failures, missed backups, silent size drops, size spikes and slow jobs. Each client is compared with **its own** history, so a 2 TB database and a 50 GB file server are each judged against their normal.
3. **Looks up** each finding in a knowledge base of error codes and procedures I wrote, by exact error code first and then by anomaly type and keywords.
4. **Explains** the night with an LLM that sees only the structured findings and the matching entries, never the raw rows.
5. **Checks** the explanation in code before showing it. If the LLM is unavailable, a code-only summary is shown instead.

On the sample report it finds all 13 planted problems, ignores the 39 harmless one-off blips, and
groups them into a prioritized handover with cited next steps.

## How it works

![Architecture](docs/architecture.svg)

| File | Role |
|---|---|
| `schema.py` | The contract every data source must meet: required columns, types, valid statuses, and which "night" a job belongs to |
| `detector.py` | Anomaly detection in pandas. Every threshold is a named constant at the top of the file |
| `kb.py` / `knowledge_base.json` | The knowledge base and its linter (checks for missing fields, duplicate codes, leftover placeholders) |
| `retriever.py` | Hybrid lookup: exact code match, then anomaly type, then BM25 keyword search |
| `summarizer.py` | The LLM call, structured output schema, guardrail checks and code-only fallback |
| `app.py` | Streamlit UI with caching and a per-session cost cap |
| `generate_backup_data.py` | Synthetic report generator that plants known anomalies and writes an answer key |
| `evaluate.py` / `retrieval_eval.py` | Scoring against the answer key and against 20 retrieval questions |

## Design decisions

**Code finds the problems. The LLM only explains them.** LLMs are unreliable at counting across
hundreds of rows. Code is exact, cheap and testable. So the detector decides what is wrong, how severe
it is and whether the night is red, amber or green. The LLM decides how to say it: grouping related
findings, ordering by urgency, and writing for an engineer reading on a phone at 7 a.m. It never sees
the raw CSV, so it can't miscount it.

**One root cause, not ten alerts.** Once a job is explained by a bigger finding (it failed because its
media agent was down), it is not reported again. The detector runs from widest cause to narrowest for
that reason.

**Every claim is checked, and the checks run in code.**

| Check | Catches |
|---|---|
| Finding IDs exist | A priority citing a finding that was never detected |
| Critical and high coverage | A serious finding left out of the summary |
| Number grounding | A number in the text that appears nowhere in the findings or retrieved entries |
| Knowledge base citations | Advice with no source, a citation to an entry that wasn't retrieved for that finding, or "not found" when entries were available |

The output must match a JSON schema enforced by the API, so the app always gets the same fields.

**"Not found" is a valid answer.** Next steps may only come from retrieved knowledge base entries. With
no match, the summary says "Not found in knowledge base." instead of inventing a fix.

**I tested embeddings and left them out.** On 20 test questions, exact code plus keyword search put the
right entry in the top 3 every time, and first 94% of the time. Adding embeddings didn't improve that,
and on their own they could not reliably say "not found": at most thresholds they returned an answer for
off-topic questions. At this size the simpler method wins, with no model download and an explanation
for every result. I would re-test as the knowledge base grows. Full numbers are in [EVALUATION.md](EVALUATION.md).

**Measured, not assumed.** The generator plants anomalies and writes an answer key the detector never
reads. I tuned thresholds on 30 generated reports, then scored 70 I had not looked at.

## Results

| What | Result |
|---|---|
| Planted anomalies caught, sample report | 13 of 13, 0 false alarms, 0 of 39 harmless blips escalated |
| Planted anomalies caught, 70 held-out reports | 907 of 910 (99.7%), 0 false alarms |
| Knowledge base lookup, 20 questions | Correct entry in top 3 for every answerable question, no false answers on 3 no-answer questions |
| LLM summary checks | Pass on the live run; each check also tested against deliberately bad output |

The three misses are slow jobs on very short backups, below the 10 extra minutes the detector requires
before it flags a duration spike. That rule is deliberate: it stops a 3-minute job that takes 12 minutes
from paging someone. Methods, limits and how to reproduce every number: [EVALUATION.md](EVALUATION.md).

## Known limitations

- **Synthetic data.** The report is generated, modeled on enterprise backup job reports. It has not been run against a production environment.
- **I wrote the test.** The detector was tuned on anomalies I designed. 99.7% on my own generator proves less than the same number on real incidents would.
- **Simple noise.** Real reports have patch nights, new clients with no history, month-end growth and retired clients. The generator doesn't model these yet, so the false-alarm rate on real data is unknown.
- **A short-job blind spot.** Slow jobs on very short backups are not flagged (see Results).
- **A small knowledge base.** 16 entries, and the retrieval test has 20 questions, mostly written by me. Real tickets are phrased differently.
- **The checks have edges.** Number grounding catches invented numbers, not real numbers used in the wrong sentence. Citation checks prove advice came from the knowledge base, not that the advice is good.
- **One measured LLM run.** The summary checks pass, but summary quality has not been rated across many runs or by other engineers.
- **Demo safeguards, not production controls.** The public app has a 5-summary-per-session cap, response caching and a monthly API spend limit. It has no login.

## Path to production

What I would change before this ran against a real environment:

1. **Data source.** Replace the CSV with the backup software's report export or REST API. Only `schema.py` needs a new adapter. The detector and everything after it stay the same.
2. **Data handling.** Client names and hostnames are internal information. Pseudonymize them before the LLM call and map them back after, or use an LLM deployment covered by the company's data agreements. Review this with security before any real data is sent.
3. **Real ground truth.** Label 30 to 60 days of past incidents from the ticketing system and score the detector against them, including the false-alarm rate on real noise.
4. **Context the detector lacks.** Feed in maintenance windows, change tickets and the CMDB, so planned work and retired clients don't raise alerts.
5. **Ownership of the knowledge base.** Each entry gets an owner and a review date, and the linter runs in CI so a broken entry can't ship.
6. **Delivery.** Run on a schedule after the backup window closes and post to the team channel or ticketing system, with the code-only summary as the fallback.
7. **Monitoring.** Log every check result, track how often summaries are flagged, and re-run the evaluation on each model or prompt change.
8. **Access.** Single sign-on in front of the app, role-based access and an audit log.
9. **Actions, carefully.** Draft tickets or suggest a rerun, with a human approving every action before it happens.

## Run it locally

Requires Python 3.11 or later.

```
git clone https://github.com/alanabreu83/backup-report-analyst
cd backup-report-analyst
python -m venv .venv
.venv\Scripts\activate             # Windows; on macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt

python generate_backup_data.py --seed 42
python detector.py backup_jobs.csv --out findings.json
python evaluate.py findings.json answer_key.json
python summarizer.py findings.json --no-llm --kb knowledge_base.json   # no API key needed
streamlit run app.py
```

For the LLM summary, put `ANTHROPIC_API_KEY=...` in a `.env` file, which is git-ignored. Never commit it.

## About

Built by Alan Abreu, a data protection architect with 20 years in enterprise backup and recovery, moving into building AI solutions. The domain knowledge
in the knowledge base, the anomaly types and the evaluation design are mine. I used Claude as a coding
partner throughout. All data is synthetic, and no employer data, code or documents were used.
