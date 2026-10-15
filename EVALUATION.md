# Evaluation

How I measured the Backup Report Analyst, what the numbers say, and where they stop meaning anything.
All data is synthetic. Every result can be reproduced with the commands at the end.

## Summary

| Part | What was measured | Result |
|---|---|---|
| Detector (code) | Planted anomalies found, false alarms, noise wrongly escalated | 13 of 13 on the sample report, 0 false alarms, 0 of 39 noise jobs escalated |
| Detector, 100 different reports | Same, on seeds I did not tune against | 99.7% on 70 held-out reports, 0 false alarms |
| Knowledge base retrieval | Right entry returned for 20 test questions | Top 3: 100% of answerable questions. Top 1: 94%. No false answers on 3 no-answer questions |
| LLM summary | Citations valid, every critical/high finding covered, numbers traceable | Checks passed on the live run. Guardrails tested against bad outputs with a fake client |

## 1. Detector

The generator plants 13 anomalies (repeat failures, a media agent outage, silent size drops, size spikes,
missed backups, slow jobs) into an 838-job report, plus 39 harmless one-off failures and warnings.
It writes an answer key that the detector never reads. `evaluate.py` compares the two.

| Report | Planted | Caught | False alarms | Noise escalated |
|---|---|---|---|---|
| Sample report (seed 42) | 13 | 13 | 0 | 0 of 39 |
| Tuning seeds 1-30 | 390 | 390 (100%) | 0 | 0 |
| Held-out seeds 31-100 | 910 | 907 (99.7%) | 0 | 0 |

I tuned thresholds on seeds 1-30 only, then scored seeds 31-100 untouched, so the held-out row is the
honest one.

**The 3 misses** (seeds 70, 71, 91) are all slow jobs that ran about 4x longer than normal. They are short
jobs, so 4x is less than the 10 extra minutes the detector requires before it flags a duration spike. That
rule is deliberate: without it, a 3-minute job that takes 12 minutes would raise an alert nobody wants. It
is a trade-off, not a bug, but it is a real blind spot.

**Limits**
- The anomalies are ones I planted and then tuned the detector to find. Real production data has anomalies
  I did not think of, and 100% on my own generator proves less than it sounds.
- Noise in the generator is simple (random one-off failures). Real noise includes patch nights, new
  clients, month-end growth and decommissioned clients, which the generator does not yet model.

## 2. Knowledge base retrieval

The knowledge base has 16 entries: 10 error-code entries and 6 procedure entries. I tested 20 questions:
17 with a correct entry, 3 with none (an off-topic question, a backup-adjacent question the knowledge base
does not cover, and an unknown error code). Questions are scored on whether the correct entry is returned
and whether the system correctly says "not found".

| Method | hit@1 | hit@3 | MRR | False answers (no-answer questions) |
|---|---|---|---|---|
| Exact code match only | 18% | 18% | 0.18 | 0 of 3 |
| Keyword search (BM25) | 76% | 82% | 0.79 | 0 of 3 |
| **Hybrid: exact code + keyword (the app)** | **94%** | **100%** | **0.97** | **0 of 3** |
| Embeddings only (similarity >= 0.7) | 76% | 82% | 0.78 | 0 of 3 |
| Hybrid + embeddings (similarity >= 0.7) | 88% | 100% | 0.92 | 0 of 3 |

hit@1 and hit@3 are how often the correct entry is first or in the top three. MRR is the average of
1 / rank of the correct entry. Embeddings used `BAAI/bge-small-en-v1.5`, run locally.

**Finding:** embeddings did not improve on the hybrid at this scale. Merging them in lowered first-place
accuracy on one question. I kept the simpler approach: no model download, no extra dependency, and every
result can be explained by which code or words matched.

**What the embedding sweep showed:** embeddings alone reached 94% hit@3 at lower similarity thresholds,
better than keyword search alone (82%). But at thresholds up to 0.55 they returned an answer for all three
no-answer questions, because a good match and a junk match score too close together. The only setting with
no false answers (0.70) dropped hit@3 to 82%. Embeddings understood meaning better but could not say "I
don't know" reliably.

**Limits**
- Small sample: one question is about 6 points of hit@1, and the no-answer set has only 3 questions.
- The similarity threshold was chosen on the same 20 questions, so the embedding rows are slightly optimistic.
- I wrote the knowledge base entries and most of the questions, so they share vocabulary. Keyword search
  benefits from that. Real tickets are phrased differently, and a larger, messier question set could change
  the ranking of these methods. I would re-run this comparison as the knowledge base grows.
- Some questions were first drafted by Claude (5 of 20, marked in the file); I rewrote the rest.

## 3. LLM summary

The summary is written by an LLM that sees only structured findings and retrieved knowledge base entries,
never the raw report. Code produces every number, and the LLM explains. Each summary is checked in code:

| Check | What it catches |
|---|---|
| Finding IDs exist | A priority that cites a finding that was never detected |
| Critical/high coverage | A serious finding left out of the summary |
| Number grounding | A number in the text that is not in the findings or retrieved entries (an invented figure) |
| Knowledge base citations | A cited entry that was not retrieved for that finding, next steps with no source, or "not found" when entries were available |

If any check fails, the app shows the warning. If the API call fails or returns invalid output, the app
falls back to a code-only summary, so it always produces something.

**Result:** on the live run for the sample report (13 findings, 8,145 input tokens, 3,293 output tokens),
all checks passed and the next steps traced to my knowledge base entries (for example the VSS writer check
under the repeat failure). The checks were also tested with a fake API client that returns deliberately bad
output: bad citations, uncited steps, invented numbers, missing fields and API errors. Each was caught.

**Limits**
- Only one real-API run is measured here. This is a check that the system works, not a statistical result
  on summary quality.
- Number grounding catches invented numbers, not real numbers used in the wrong place.
- The checks confirm the advice came from the knowledge base. They do not confirm the advice is good, which
  depends on the entries.

## Reproduce

```
python generate_backup_data.py --seed 42            # sample data and answer key
python detector.py backup_jobs.csv --out findings.json
python evaluate.py findings.json answer_key.json    # detector score
python retrieval_eval.py                            # retrieval, keyword methods
pip install fastembed
python retrieval_eval.py --embedder fastembed       # adds the embedding comparison
python summarizer.py findings.json --kb knowledge_base.json   # needs an API key
```
