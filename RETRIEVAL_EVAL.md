## Retrieval evaluation

Retrieval evaluation: 20 questions (5 by claude, 15 by alan), 16 KB entries, embedder BAAI/bge-small-en-v1.5

| Method | hit@1 | hit@3 | MRR | False answers on no-answer questions |
|---|---|---|---|---|
| exact | 18% | 18% | 0.18 | 0% (0 of 3) |
| keyword (BM25) | 76% | 82% | 0.79 | 0% (0 of 3) |
| hybrid (today's app) | 94% | 100% | 0.97 | 0% (0 of 3) |
| embeddings only (>= 0.7) | 76% | 82% | 0.78 | 0% (0 of 3) |
| hybrid + embeddings (>= 0.7) | 88% | 100% | 0.92 | 0% (0 of 3) |

Embedding similarity threshold used: 0.7
