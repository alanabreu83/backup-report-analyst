"""
retriever.py: hybrid search over the knowledge base.

Three signals, combined into one ranking:
    1. EXACT code match   - "Error Code: 101" in a finding equals an entry's code. This is
                            a lookup, not a guess, so it always outranks everything else.
    2. TYPE match         - the finding's type (size_drop, ...) equals a procedure entry's type.
    3. KEYWORD (BM25)     - word overlap between the query and the entry text. Catches
                            free-text questions and near-misses the first two can't.

Why not embeddings yet? Error codes are exact identifiers; keyword + exact match handles
them better than semantic search. Embeddings are added later as a measured comparison
(Days 9-10) so the choice is backed by numbers, not assumed.

BM25 is written by hand here (about 20 lines) so there is nothing to install and every
score can be explained.
"""

import math
import re
from collections import Counter

from kb import entry_text

CODE_BOOST = 100.0       # exact code match always wins
TYPE_BOOST = 20.0        # procedure for this kind of anomaly
MIN_KEYWORD_SCORE = 1.5  # keyword-only hits below this are treated as "not found"
K1, B = 1.5, 0.75

_STOP = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
         "of", "on", "or", "that", "the", "to", "was", "with", "this", "not", "no", "has", "have",
         "what", "when", "how", "why", "who", "which", "do", "does", "did", "if", "there", "can",
         "will", "you", "your", "i", "we", "my", "me", "should"}
_WORD = re.compile(r"[a-z0-9]+")
_CODE_PREFIX = re.compile(r"^\s*(error\s*code|errorcode|code|err)\b[\s:#=-]*", re.I)


def normalize_code(code):
    """'Error Code: 101', 'error code 101', 'ERR-101' and '101' all become '101'.
    'SIM-1002' becomes 'SIM1002'. Only letters and digits survive."""
    s = _CODE_PREFIX.sub("", str(code))
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def tokenize(text):
    tokens = []
    for w in _WORD.findall(str(text).lower()):
        if w in _STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # crude plural stripping: "snapshots" -> "snapshot"
        tokens.append(w)
    return tokens


class Retriever:
    def __init__(self, entries):
        self.entries = entries
        self.docs = [Counter(tokenize(entry_text(e))) for e in entries]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(entries)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.by_code = {}
        for e in entries:
            for c in e["error_codes"]:
                self.by_code.setdefault(normalize_code(c), []).append(e)

    def _bm25(self, i, query_tokens):
        """Return (score, number of distinct query words found in the entry)."""
        doc, length = self.docs[i], self.lengths[i]
        score, matched = 0.0, 0
        for t in set(query_tokens):
            tf = doc.get(t, 0)
            if not tf:
                continue
            matched += 1
            norm = tf + K1 * (1 - B + B * length / (self.avg_len or 1))
            score += self.idf[t] * tf * (K1 + 1) / norm
        return score, matched

    def search(self, query="", codes=(), finding_type=None, k=3, require_signal=False):
        """Return up to k hits: [{'entry', 'score', 'why'}], best first. May be empty.
        require_signal=True means keywords may only re-rank entries that already matched by
        code or finding type (used for findings, whose text is machine-written and noisy)."""
        qtokens = tokenize(query)
        wanted = {normalize_code(c) for c in codes if str(c).strip()}
        scored = []
        for i, e in enumerate(self.entries):
            score, reasons = 0.0, []
            if wanted & {normalize_code(c) for c in e["error_codes"]}:
                score += CODE_BOOST
                reasons.append("exact code")
            if finding_type and finding_type in e["finding_types"]:
                score += TYPE_BOOST
                reasons.append("finding type")
            kw, matched = self._bm25(i, qtokens) if qtokens else (0.0, 0)
            # A keyword-only hit needs a decent score AND (for multi-word queries) at least two
            # different words in common, so one stray shared word can't produce an answer.
            enough = matched >= min(2, len(set(qtokens)))
            if (kw >= MIN_KEYWORD_SCORE and enough and not require_signal) or (reasons and kw > 0):
                score += kw
                reasons.append("keyword")
            if reasons:
                scored.append({"entry": e, "score": round(score, 2), "why": " + ".join(reasons)})
        scored.sort(key=lambda h: (-h["score"], h["entry"]["id"]))
        return scored[:k]


def retrieve_for_finding(finding, retriever, k=3):
    codes = list((finding.get("evidence") or {}).get("error_codes", {}).keys())
    query = finding["type"].replace("_", " ")
    return retriever.search(query, codes=codes, finding_type=finding["type"], k=k, require_signal=True)


def retrieve_for_result(result, retriever, k=3):
    """Hits for every finding: {finding_id: [hit, ...]}. The KB lookup is plain code, no LLM."""
    return {f["id"]: retrieve_for_finding(f, retriever, k) for f in result["findings"]}
