#!/usr/bin/env python3
"""
retrieval_eval.py: does the knowledge base lookup return the right entry?

Runs each test question through several retrieval methods and scores them:

    exact          only error codes found in the question text
    keyword        BM25 word matching only
    hybrid         exact code + finding type + keyword (what the app uses today)
    embed          embeddings only (needs --embedder)
    hybrid+embed   hybrid and embeddings merged (needs --embedder)

Scoring (questions that have an answer in the KB):
    hit@1   the correct entry is the top result
    hit@3   the correct entry is in the top 3
    MRR     mean of 1/rank of the first correct entry (1.0 = always first)
Scoring (questions that should find NOTHING, such as off-topic ones):
    false answer rate   how often the method returned something anyway.
    Returning a confident wrong entry is worse than saying "not found".

Usage:
    python retrieval_eval.py                              # keyword methods only
    python retrieval_eval.py --embedder fastembed         # adds the embedding methods
    python retrieval_eval.py --embedder voyage            # needs VOYAGE_API_KEY
    python retrieval_eval.py --embedder fastembed --min-sim 0.6

Writes retrieval_results.json and RETRIEVAL_EVAL.md (tables for your README).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from kb import DEFAULT_KB_PATH, entry_text, load_kb, usable_entries
from retriever import Retriever, normalize_code

HERE = Path(__file__).parent
SWEEP = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
RRF_K = 60


def resolve_expected(entries, expect):
    """Turn {'codes': [...], 'types': [...]} into the set of correct entry IDs."""
    ids = set()
    codes = {normalize_code(c) for c in expect.get("codes", [])}
    types = set(expect.get("types", []))
    for e in entries:
        if codes & {normalize_code(c) for c in e["error_codes"]}:
            ids.add(e["id"])
        if types & set(e["finding_types"]):
            ids.add(e["id"])
    return ids


def rank_of_first_hit(ranked_ids, expected, k):
    for i, eid in enumerate(ranked_ids[:k], start=1):
        if eid in expected:
            return i
    return None


class Methods:
    def __init__(self, entries, embedder=None, min_sim=0.6):
        self.entries = entries
        self.retriever = Retriever(entries)
        self.embedder, self.min_sim = embedder, min_sim
        if embedder:
            self.doc_vecs = embedder.embed_documents([entry_text(e) for e in entries])
        self.query_cache = {}

    def exact(self, q, k):
        codes = self.retriever.find_codes(q)
        hits = self.retriever.search("", codes=codes, k=k) if codes else []
        return [h["entry"]["id"] for h in hits]

    def keyword(self, q, k):
        return [h["entry"]["id"] for h in self.retriever.search(q, k=k)]

    def hybrid_hits(self, q, k):
        return self.retriever.search(q, k=k, extract_codes=True)

    def hybrid(self, q, k):
        return [h["entry"]["id"] for h in self.hybrid_hits(q, k)]

    def sims(self, q):
        if q not in self.query_cache:
            self.query_cache[q] = (self.doc_vecs @ self.embedder.embed_queries([q])[0])
        return self.query_cache[q]

    def embed(self, q, k, min_sim=None):
        thr = self.min_sim if min_sim is None else min_sim
        s = self.sims(q)
        order = np.argsort(-s)
        return [self.entries[i]["id"] for i in order[:k] if s[i] >= thr]

    def fused(self, q, k, min_sim=None):
        """Reciprocal rank fusion of the hybrid list and the embedding list.
        An exact code match is kept on top, because a code lookup is not a guess."""
        scores = Counter()
        for rank, h in enumerate(self.hybrid_hits(q, 10), start=1):
            scores[h["entry"]["id"]] += 1 / (RRF_K + rank) + (1.0 if "exact code" in h["why"] else 0)
        for rank, eid in enumerate(self.embed(q, 10, min_sim), start=1):
            scores[eid] += 1 / (RRF_K + rank)
        return [eid for eid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))][:k]


def score(method, questions, entries, k=3, kmax=5):
    inside, outside = [], []
    per_question = []
    for q in questions:
        expected = resolve_expected(entries, q["expect"])
        ranked = method(q["q"], kmax)
        if q["expect"] and expected:
            r = rank_of_first_hit(ranked, expected, kmax)
            inside.append(r)
            per_question.append({"id": q["id"], "q": q["q"], "expected": sorted(expected),
                                 "got": ranked[:k], "rank": r})
        elif not q["expect"]:
            outside.append(bool(ranked[:k]))
            per_question.append({"id": q["id"], "q": q["q"], "expected": [], "got": ranked[:k],
                                 "rank": None})
    n = len(inside)
    return {
        "n_answerable": n,
        "hit@1": sum(r == 1 for r in inside) / n if n else None,
        "hit@3": sum(r is not None and r <= 3 for r in inside) / n if n else None,
        "mrr": sum(1 / r for r in inside if r) / n if n else None,
        "n_unanswerable": len(outside),
        "false_answer_rate": sum(outside) / len(outside) if outside else None,
        "per_question": per_question,
    }


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def table(results):
    lines = ["| Method | hit@1 | hit@3 | MRR | False answers on no-answer questions |",
             "|---|---|---|---|---|"]
    for name, r in results.items():
        mrr = "n/a" if r["mrr"] is None else f"{r['mrr']:.2f}"
        far = "n/a" if r["false_answer_rate"] is None else \
            f"{r['false_answer_rate']:.0%} ({round(r['false_answer_rate'] * r['n_unanswerable'])} of {r['n_unanswerable']})"
        lines.append(f"| {name} | {pct(r['hit@1'])} | {pct(r['hit@3'])} | {mrr} | {far} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Score knowledge base retrieval.")
    p.add_argument("--kb", default=str(DEFAULT_KB_PATH))
    p.add_argument("--questions", default=str(HERE / "retrieval_questions.json"))
    p.add_argument("--embedder", choices=["fastembed", "voyage", "hash"])
    p.add_argument("--embed-model")
    p.add_argument("--min-sim", type=float, help="Minimum similarity for an embedding hit "
                   "(default: the best value from the sweep)")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()

    entries = usable_entries(load_kb(args.kb))
    questions = json.load(open(args.questions, encoding="utf-8"))["questions"]
    authors = Counter(q.get("author", "?") for q in questions)

    unresolved = [q["id"] for q in questions if q["expect"] and not resolve_expected(entries, q["expect"])]
    if unresolved:
        print(f"Skipping {len(unresolved)} questions whose expected entry is not in the KB "
              f"(fix the code/type in the questions file, or write the entry): {', '.join(unresolved)}\n")

    embedder = None
    if args.embedder:
        from embedder import make_embedder
        embedder = make_embedder(args.embedder, args.embed_model)
    m = Methods(entries, embedder)
    k = args.k

    sweep_rows, chosen = [], None
    if embedder:
        for thr in SWEEP:
            r_e = score(lambda q, kk, t=thr: m.embed(q, kk, t), questions, entries, k)
            r_f = score(lambda q, kk, t=thr: m.fused(q, kk, t), questions, entries, k)
            sweep_rows.append((thr, r_e, r_f))
        # best = highest hit@3 for the merged method, then fewest false answers, then lowest threshold
        chosen = max(sweep_rows, key=lambda t: ((t[2]["hit@3"] or 0) - (t[2]["false_answer_rate"] or 0), -t[0]))[0]
        if args.min_sim is None:
            args.min_sim = chosen
        m.min_sim = args.min_sim

    results = {
        "exact": score(m.exact, questions, entries, k),
        "keyword (BM25)": score(m.keyword, questions, entries, k),
        "hybrid (today's app)": score(m.hybrid, questions, entries, k),
    }
    if embedder:
        results[f"embeddings only (>= {m.min_sim})"] = score(m.embed, questions, entries, k)
        results[f"hybrid + embeddings (>= {m.min_sim})"] = score(m.fused, questions, entries, k)

    head = (f"Retrieval evaluation: {len(questions)} questions "
            f"({', '.join(f'{n} by {a}' for a, n in authors.items())}), {len(entries)} KB entries"
            + (f", embedder {embedder.name}" if embedder else ", keyword methods only"))
    print(head + "\n")
    print(table(results))

    if sweep_rows:
        print("\nSimilarity threshold sweep (higher = stricter; trades missed answers for false answers)")
        print("| min similarity | embeddings hit@3 | embeddings false answers | hybrid+embed hit@3 | hybrid+embed false answers |")
        print("|---|---|---|---|---|")
        for thr, e, f in sweep_rows:
            star = "  <- best" if thr == chosen else ""
            print(f"| {thr:.2f} | {pct(e['hit@3'])} | {pct(e['false_answer_rate'])} | "
                  f"{pct(f['hit@3'])} | {pct(f['false_answer_rate'])} |{star}")
        print("\nNote: the threshold was picked on these same questions, so the merged-method numbers "
              "are a little optimistic. Add questions you did not tune on before you quote them.")

    best_name = list(results)[-1] if embedder else "hybrid (today's app)"
    misses = [q for q in results[best_name]["per_question"]
              if (q["expected"] and (q["rank"] is None or q["rank"] > k)) or (not q["expected"] and q["got"])]
    print(f"\nMisses for: {best_name}")
    for q in misses:
        want = ", ".join(q["expected"]) or "nothing"
        print(f"  {q['id']}  wanted {want}; got {', '.join(q['got']) or 'nothing'}  |  {q['q']}")
    if not misses:
        print("  none")

    out = Path(args.out_dir)
    (out / "retrieval_results.json").write_text(json.dumps(
        {"head": head, "results": results, "sweep": [
            {"min_sim": t, "embed": e, "hybrid+embed": f} for t, e, f in sweep_rows]}, indent=2, default=float),
        encoding="utf-8")
    md = [f"## Retrieval evaluation\n\n{head}\n", table(results)]
    if sweep_rows:
        md.append(f"\nEmbedding similarity threshold used: {m.min_sim}")
    (out / "RETRIEVAL_EVAL.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nWrote retrieval_results.json and RETRIEVAL_EVAL.md")


if __name__ == "__main__":
    main()
