"""
embedder.py: turns text into vectors, so "snapshot would not complete" can sit close to
"VSS snapshot creation failed" even though they share almost no words.

Used only by retrieval_eval.py, to measure whether embeddings beat keyword search on
THIS knowledge base. Nothing here is imported by the app.

Backends:
    fastembed  Runs on your own computer. No account, no API key, no data leaves the machine.
               pip install fastembed      (first run downloads a ~130 MB model)
    voyage     Hosted API. Anthropic has no embedding model of its own and points to Voyage AI.
               pip install voyageai       and set VOYAGE_API_KEY (needs a Voyage/MongoDB account)
    hash       Test stand-in built from word hashes. NOT semantic; for checking the plumbing only.
"""

import os
import re

import numpy as np

DEFAULT_MODELS = {"fastembed": "BAAI/bge-small-en-v1.5", "voyage": "voyage-4-lite", "hash": "hash-256"}


def _normalize(m):
    m = np.asarray(m, dtype="float32")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class FastEmbedder:
    def __init__(self, model=None):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise SystemExit("fastembed is not installed. Run: pip install fastembed")
        self.name = model or DEFAULT_MODELS["fastembed"]
        self.model = TextEmbedding(self.name)

    def embed_documents(self, texts):
        return _normalize(list(self.model.embed(list(texts))))

    def embed_queries(self, texts):
        # BGE models are trained to expect a short instruction on the question side
        prefix = "Represent this sentence for searching relevant passages: "
        return _normalize(list(self.model.embed([prefix + t for t in texts])))


class VoyageEmbedder:
    def __init__(self, model=None):
        try:
            import voyageai
        except ImportError:
            raise SystemExit("voyageai is not installed. Run: pip install voyageai")
        if not os.environ.get("VOYAGE_API_KEY"):
            raise SystemExit("Set VOYAGE_API_KEY first (this key is separate from your Anthropic key).")
        self.name = model or DEFAULT_MODELS["voyage"]
        self.client = voyageai.Client()

    def _embed(self, texts, input_type):
        return _normalize(self.client.embed(list(texts), model=self.name, input_type=input_type).embeddings)

    def embed_documents(self, texts):
        return self._embed(texts, "document")

    def embed_queries(self, texts):
        return self._embed(texts, "query")


class HashEmbedder:
    """Bag-of-words hashed into 256 buckets. Deterministic, offline, and NOT semantic."""
    name = "hash-256"

    def _embed(self, texts):
        m = np.zeros((len(texts), 256), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                m[i, sum(ord(c) * 31 ** j for j, c in enumerate(w)) % 256] += 1
        return _normalize(m)

    embed_documents = embed_queries = _embed


def make_embedder(kind, model=None):
    if kind == "fastembed":
        return FastEmbedder(model)
    if kind == "voyage":
        return VoyageEmbedder(model)
    if kind == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown embedder: {kind}")
